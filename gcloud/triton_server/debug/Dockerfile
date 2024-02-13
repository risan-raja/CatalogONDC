# Use the official NVIDIA Triton Inference Server image as the base image
FROM nvcr.io/nvidia/tritonserver:24.01-py3
COPY 99-check-run-aip-mode.sh /opt/nvidia/entrypoint.d/99-check-run-aip-mode.sh
# Set an environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS=/etc/application-credentials.json