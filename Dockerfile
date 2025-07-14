# A Dockerfile is a text file that contains a set of instructions for Docker to build a Docker image.

# base image
FROM python:3.11

# Install system tools/dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

#if it prints tesseract version then it means it successfully installed (See in logs)
RUN tesseract --version

# Set working directory
WORKDIR /app

# Copy project files (Copies your code into the image)
COPY . /app

# Remove known incompatible packages before install as they are windows specific only (removed to avoid install errors on Linux.)
RUN sed -i '/pywin32/d' requirements.txt && \
    sed -i '/pyreadline3/d' requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#image created now below this is for starting container from the image

# Expose Streamlit port
EXPOSE 8501

# Start app (below is command to run app)
CMD sh -c "streamlit run ResearchGPT_withAgenticAI_Streamlit_v6.py --server.port=$PORT --server.address=0.0.0.0"

