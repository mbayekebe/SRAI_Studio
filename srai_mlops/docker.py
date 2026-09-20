from pathlib import Path
def generate_dockerfile(base_image="python:3.11-slim",requirements_file="requirements.txt",app_module="app:app",port=8000):
    return f"""FROM {base_image}
WORKDIR /app
COPY {requirements_file} /app/{requirements_file}
RUN pip install --no-cache-dir -r {requirements_file}
COPY . /app
EXPOSE {port}
CMD ["uvicorn", "{app_module}", "--host", "0.0.0.0", "--port", "{port}"]
"""
def generate_dockerignore():
    return "__pycache__/\n*.pyc\n.ipynb_checkpoints/\n.git/\n.env\nartifacts/\ndata/raw/\n"
def image_size_estimate(base_mb,dependency_mb,artifact_mb):
    return float(base_mb+dependency_mb+artifact_mb)
def write_container_files(directory,dockerfile_text,dockerignore_text=None):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    dockerfile=directory/"Dockerfile"; dockerfile.write_text(dockerfile_text,encoding="utf-8")
    dockerignore=directory/".dockerignore"; dockerignore.write_text(dockerignore_text or generate_dockerignore(),encoding="utf-8")
    return dockerfile,dockerignore
