import os
import streamlit as st
from google import genai
from google.genai import types


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="centered",
)

st.title("📄 AI Resume Analyzer")
st.write(
    "Upload your resume, paste a job description, and get AI-powered "
    "feedback on how well your resume matches the job."
)

st.info(
    "Note: This tool provides AI-generated career feedback. "
    "It is not a guarantee of ATS or hiring results."
)


# -----------------------------
# Gemini client
# -----------------------------
def get_api_key():
    """Get the Gemini API key from Streamlit Secrets or an environment variable."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None

    if not key:
        key = os.getenv("GEMINI_API_KEY")

    return key


def get_gemini_client():
    """Create a Gemini client using the user's API key."""
    api_key = get_api_key()

    if not api_key:
        st.error(
            "Gemini API key not found. Add GEMINI_API_KEY to "
            "Streamlit Secrets before using the analyzer."
        )
        st.stop()

    return genai.Client(api_key=api_key)


# -----------------------------
# Resume analysis
# -----------------------------
def analyze_resume(resume_file, job_description):
    """Send the PDF resume and job description to Gemini."""
    client = get_gemini_client()

    resume_bytes = resume_file.getvalue()

    prompt = f"""
You are an expert resume reviewer and ATS-focused career assistant.

Analyze the uploaded PDF resume against the job description below.

JOB DESCRIPTION:
{job_description}

Give the user a practical and honest analysis.

Use this exact structure:

OVERALL MATCH SCORE
Give a score from 0 to 100 and briefly explain it.

MATCHING SKILLS
List the important skills, qualifications, tools, or experience that match.

MISSING OR WEAK AREAS
List important job requirements that are missing, unclear, or weak in the resume.
Do not invent experience that is not present in the resume.

RESUME STRENGTHS
List the strongest parts of the resume for this particular job.

IMPROVEMENTS
Give specific suggestions for improving the resume for this job.

ATS KEYWORDS
List relevant keywords from the job description that the resume should naturally include
if they are truthful and supported by the candidate's actual experience.

ACTION PLAN
Give 3 to 5 practical next steps.

Important rules:
- Only use information actually present in the resume and job description.
- Never invent degrees, jobs, skills, certifications, or achievements.
- Do not discriminate based on age, gender, religion, nationality, race, disability,
  or other protected characteristics.
- Keep the feedback professional, clear, and beginner-friendly.
- Do not guarantee employment or a specific ATS result.
"""

    try:
        pdf_part = types.Part.from_bytes(
            data=resume_bytes,
            mime_type="application/pdf",
        )

        response = client.models.generate_content(
            model="openai/gpt-oss-120b",
            contents=[pdf_part, prompt],
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text

    except Exception as error:
        raise RuntimeError(f"Gemini API error: {error}") from error


# -----------------------------
# User interface
# -----------------------------
resume_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"],
)

job_description = st.text_area(
    "Paste the job description",
    height=250,
    placeholder="Paste the complete job description here...",
)

analyze_button = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True,
)

if analyze_button:
    if resume_file is None:
        st.warning("Please upload your resume PDF first.")
    elif not job_description.strip():
        st.warning("Please paste a job description first.")
    elif resume_file.size == 0:
        st.warning("The uploaded PDF appears to be empty.")
    else:
        with st.spinner("Analyzing your resume..."):
            try:
                result = analyze_resume(resume_file, job_description)

                st.success("Analysis complete!")
                st.markdown("## 📊 Resume Analysis")
                st.markdown(result)

            except Exception as error:
                st.error(
                    "The analysis could not be completed. "
                    "Please check your API key, internet connection, "
                    "PDF file, and Gemini usage limits."
                )
                st.exception(error)
