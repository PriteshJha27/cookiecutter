# Environment variables and credentials
.env
*.env
.env.*
!.env.example

# Certificates and sensitive files
*.crt
*.key
*.pem
*.cer
/amexcerts/
**/certificate_path/

# Authentication tokens and secrets
**/tokens/
*_token
*_secret
*.token
*.secret

# Credentials and config files with sensitive data
credentials.yaml
credentials.json
secrets.yaml
secrets.json

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
.venv/
.env/
env.bak/
venv.bak/

# IDE specific files
.idea/
.vscode/
*.swp
*.swo
*~
.project
.classpath
.settings/
*.sublime-workspace
*.sublime-project

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs and databases
*.log
*.sqlite
*.db
logs/
log/
*.log.*
crash-*

# Model files and large binary files
*.bin
*.h5
*.pkl
*.model
models/
/minilm/
*.onnx

# Vector store files and indexes
vector_store/
indexes/
*.index
faiss.db
*.faiss
datastax/

# Temporary files
tmp/
temp/
.temp/
.tmp/

# Generated files
*.generated.*
generated/
dist/
build/

# Project specific
data/*
!data/.gitkeep
storage/*
!storage/.gitkeep
results/*
!results/.gitkeep
safechain_*/

# Notebooks checkpoints
.ipynb_checkpoints/
*/.ipynb_checkpoints/*
*.ipynb
!examples/*.ipynb

# Test coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/
