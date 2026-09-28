# Clinical Q&A Assistant

RAG-based clinical document Q&A. Flask + ReactJS interface, FAISS for vector
search, MongoDB for document/chunk/query storage, Gemini for embeddings and
generation.

# 3. Push this project to GitHub

```bash
cd clinical-qa-app
git init
git add .
git commit -m "Initial commit"
gh repo create clinical-qa-assistant --public --source=. --push
# (or create a repo on github.com and `git remote add origin <url> && git push -u origin main`)
```

# Local development (optional, before deploying)

Backend:
```bash
cd backend
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # then fill in MONGODB_URI and GEMINI_API_KEY
python app.py          # runs on http://localhost:5000
```

Frontend (in a second terminal):
```bash
cd frontend
npm install
REACT_APP_API_BASE=http://localhost:5000 npm start   # runs on http://localhost:3000
```
