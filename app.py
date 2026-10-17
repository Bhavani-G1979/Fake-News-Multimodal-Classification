import os

# Load secrets from .streamlit/secrets.toml before any library imports that
# might use API keys for model downloads.
_SECRETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             ".streamlit", "secrets.toml")
if os.path.exists(_SECRETS_PATH):
    try:
        import toml
        with open(_SECRETS_PATH) as _f:
            _secrets = toml.load(_f)
        for _k, _v in _secrets.items():
            os.environ.setdefault(_k, _v)
    except Exception:
        pass

import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from openai import (
    OpenAI,
    OpenAIError,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
)
from PIL import Image, ImageOps
from torchvision.transforms import v2
from transformers import BertTokenizer

from Model import BERTResNetClassifier

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "TextImagelatest.pth")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CLASS_LABELS = [
    "TRUE",
    "SATIRE",
    "FALSE CONNECTION",
    "IMPOSTER CONTENT",
    "MANIPULATED CONTENT",
    "MISLEADING CONTENT",
]

CLASS_DESCRIPTIONS = [
    "Content that is factually accurate and based on verified information.",
    "Content created for humorous or satirical purposes, often not meant to be taken seriously.",
    "Headlines or visuals that mislead or do not accurately reflect the content of the article.",
    "Content that impersonates genuine sources, such as fake news articles mimicking legitimate media outlets.",
    "Content that has been altered, edited, or manipulated to mislead or distort facts.",
    "Content that uses selective information or framing to mislead the audience, often by omitting context.",
]


@st.cache_resource
def load_model():
    model = BERTResNetClassifier(load_pretrained_weights=False)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
    )
    model.eval()
    return model


@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


def get_bert_embedding(text, tokenizer):
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=80,
        truncation=True,
        padding="max_length",
    )
    return inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0)


def get_openai_api_key():
    """Read the OpenAI API key from Streamlit secrets first, then env var fallback."""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return key.strip()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "").strip()


def get_openai_assessment(headline, image):
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to `.streamlit/secrets.toml` "
            "or set it as an environment variable to enable the AI second opinion."
        )

    image_buffer = io.BytesIO()
    image.thumbnail((1024, 1024))
    image.save(image_buffer, format="JPEG", quality=85, optimize=True)
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        max_tokens=250,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Assess the relationship between the supplied news headline and image "
                    "using only the six provided content categories. You cannot establish "
                    "whether a claim is factually true from this input alone. If evidence "
                    "is insufficient, set label to UNCERTAIN. Return a JSON object with "
                    "exactly these fields: label (one exact category name or UNCERTAIN), "
                    "uncertain (boolean), rationale (one or two brief sentences)."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Categories: {', '.join(CLASS_LABELS)}. "
                            f"Headline: {headline}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("The AI service returned an empty assessment.")

    assessment = json.loads(content)
    if not isinstance(assessment, dict):
        raise ValueError("The AI service returned an assessment in an invalid format.")
    label = assessment.get("label")
    uncertain = assessment.get("uncertain")
    rationale = assessment.get("rationale")
    if (
        label not in CLASS_LABELS + ["UNCERTAIN"]
        or not isinstance(uncertain, bool)
        or not isinstance(rationale, str)
        or not rationale.strip()
    ):
        raise ValueError("The AI service returned an assessment in an invalid format.")
    if uncertain:
        label = "UNCERTAIN"

    return {"label": label, "uncertain": uncertain, "rationale": rationale.strip()}


st.set_page_config(
    page_title="Fakeddit App",
    page_icon="📰",
    layout="centered",
)

st.title("Fakeddit App")
st.caption(
    "Predictions are model assessments, not fact-checks. Use the optional AI second "
    "opinion for an additional perspective."
)

uploaded_title = st.text_input("Article Headline", placeholder="Enter an article headline")
uploaded_file = st.file_uploader(
    "Choose an accompanying image for the article...", type=["jpg", "jpeg", "png"]
)
use_openai = st.checkbox(
    "Get an OpenAI second opinion",
    help="When enabled, the headline and a resized copy of the image are sent to OpenAI.",
)
if use_openai:
    st.caption(
        "Privacy notice: your headline and image will be sent to OpenAI when you click "
        "Predict. This assessment is not a substitute for fact-checking."
    )

if uploaded_file is not None:
    input_image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    st.image(input_image, caption="Uploaded Image Preview", width="stretch")

if st.button("Predict"):
    if uploaded_file is None:
        st.text("Please upload an image first.")
    elif not uploaded_title.strip():
        st.text("Please enter an article headline.")
    else:
        try:
            with st.spinner("Loading model and tokenizer..."):
                model = load_model()
                tokenizer = load_tokenizer()

            with st.spinner("Analyzing..."):
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

                transform_func = v2.Compose(
                    [
                        v2.Resize((256, 256)),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean, std),
                    ]
                )

                input_ids, attention_mask = get_bert_embedding(uploaded_title, tokenizer)
                uploaded_file.seek(0)
                input_image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
                img_tensor = transform_func(input_image).unsqueeze(0)

                with torch.inference_mode():
                    output = model(
                        image=img_tensor,
                        text_input_ids=input_ids.unsqueeze(0),
                        text_attention_mask=attention_mask.unsqueeze(0),
                    )

            probabilities = F.softmax(output, dim=1).squeeze()

            predicted_class = torch.argmax(output, dim=1).item()
            predicted_label = CLASS_LABELS[predicted_class]

            st.subheader(f"Local model prediction: {predicted_label}")
            st.text(CLASS_DESCRIPTIONS[predicted_class])
            st.text(f"Confidence: {probabilities[predicted_class]:.2%}")

            st.markdown("---")
            st.caption("Local model class probabilities:")
            prob_df = pd.DataFrame({
                "Label": CLASS_LABELS,
                "Probability": [f"{p:.2%}" for p in probabilities.tolist()],
            })
            st.table(prob_df)

            if use_openai:
                uploaded_file.seek(0)
                with Image.open(uploaded_file) as original_image:
                    ai_image = ImageOps.exif_transpose(original_image).convert("RGB")
                try:
                    with st.spinner("Getting the OpenAI second opinion..."):
                        assessment = get_openai_assessment(uploaded_title, ai_image)
                    st.subheader(f"OpenAI second opinion: {assessment['label']}")
                    st.write(assessment["rationale"])
                    if assessment["uncertain"]:
                        st.info("The AI indicated that there is not enough evidence to classify this item.")
                except AuthenticationError:
                    st.error(
                        "The OpenAI API key is invalid or expired. "
                        "Please check your key in `.streamlit/secrets.toml`."
                    )
                except RateLimitError:
                    st.error(
                        "OpenAI rate limit exceeded. Please try again later "
                        "or check your usage limits at https://platform.openai.com/usage."
                    )
                except APITimeoutError:
                    st.error("OpenAI API request timed out. Please try again later.")
                except APIConnectionError:
                    st.error(
                        "Could not connect to OpenAI. "
                        "Please check your network connection and try again."
                    )
                except APIStatusError as error:
                    st.error(f"OpenAI API error (status {error.status_code}): {error.message}")
                except OpenAIError as error:
                    st.error(f"OpenAI API error: {error}")
                except ValueError as error:
                    st.error(f"OpenAI second opinion error: {error}")
                except (json.JSONDecodeError, IndexError, TypeError) as error:
                    st.error(f"Could not parse the OpenAI response: {error}")
                except Exception as error:
                    st.error(f"An unexpected error occurred: {error}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
