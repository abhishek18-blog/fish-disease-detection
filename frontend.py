import streamlit as st
import requests, os
from PIL import Image
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Smart Fish Diagnosis", layout="wide")
st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #0066cc;
        border-color: #0066cc;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0052a3;
        border-color: #0052a3;
    }
    </style>
""", unsafe_allow_html=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODELS = ["EfficientNet-B0", "MobileNetV2", "DenseNet-121", "ResNet-50"]
report = []

def get_ai_explanation(results_summary):
    prompt = f"""
    You are an expert Aquatic Veterinarian. A fish disease diagnostic system has analyzed an image 
    using four different AI architectures. Here are the results:
    {results_summary}

    Based on these consensus or conflicting results:
    1. Determine the most likely diagnosis.
    2. Explain what this disease is in simple terms.
    3. Provide 3-4 clear, actionable treatment steps (quarantine, medication, water changes, etc.).
    
    Keep the tone professional, supportive, and concise.
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content

left_col, mid_col, right_col = st.columns([1, 1.5, 1], gap="large")

with left_col:
    st.title("Smart Fish Diagnosis")

    with st.container(border=True):
        st.markdown("Upload a fish image to get a health diagnosis.")
        uploaded_file = st.file_uploader("Upload Fish Image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")
            run_diagnosis = st.button("Run Diagnosis", use_container_width=True, type="primary")
    
with mid_col:
    st.header("Diagnostic Report")
    
    if uploaded_file is None:
        st.info("Please upload an image on the left and click 'Run Diagnosis' to view the report.")
        
    elif 'run_diagnosis' in locals() and run_diagnosis:
        st.divider()
        
        row1_cols = st.columns(2)
        row2_cols = st.columns(2)
        grid_cols = row1_cols + row2_cols 
        
        file_bytes = uploaded_file.getvalue() 
        
        for index, model_name in enumerate(MODELS):
            with grid_cols[index]:
                with st.container(border=True):
                    st.markdown(f"#### {model_name}")
                    
                    with st.spinner("Analyzing..."):
                        files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                        data = {"model_selection": model_name}
                        
                        try:
                            response = requests.post("http://127.0.0.1:8000/predict", files=files, data=data)
                            
                            if response.status_code == 200:
                                result = response.json()
                                diag = result["diagnosis"]
                                conf = result["confidence"]
                                
                                st.caption("Predicted Disease")
                                st.markdown(f"### {diag}")
                                st.metric("Confidence Score", conf)
                                
                                report.append(f"Model {model_name} predicted {diag} with {conf} confidence.")
                                
                                if diag == "Healthy Fish":
                                    st.success("Status: Healthy")
                                else:
                                    st.error("Status: Alert")
                                    
                            else:
                                st.error(f"API Error: {response.status_code}")
                                
                        except requests.exceptions.ConnectionError:
                            st.error("Backend Offline. Ensure FastAPI is running.")

with right_col:
    st.header("AI Analysis")
    
    if len(report) == 0:
        st.info("Run Diagnosis to get Analysis.")
    else:
        with st.expander("💬 View AI Analysis"):
            with st.spinner("Analyzing..."):
                summary_text = "\n".join(report)
                ai_response = get_ai_explanation(summary_text)
                
                st.markdown(ai_response)
                st.caption("Disclaimer: AI diagnosis should be verified by an expert.")