# Clinical Q&A Assistant

RAG-based clinical document Q&A. Flask + ReactJS interface, FAISS for vector
search, MongoDB for document/chunk/query storage, Gemini for embeddings and
generation.

## 3. Push this project to GitHub

```bash
cd clinical-qa-app
git init
git add .
git commit -m "Initial commit"
gh repo create clinical-qa-assistant --public --source=. --push
# (or create a repo on github.com and `git remote add origin <url> && git push -u origin main`)
```

## 4. Deploy on Render (15-20 min)

1. Go to https://dashboard.render.com -> **New** -> **Blueprint**.
2. Connect your GitHub account and pick this repo. Render will read `render.yaml`
   automatically and propose the `clinical-qa-assistant` web service.
3. Before the first deploy, set the two secret environment variables it asks for:
   - `MONGODB_URI` -> the connection string from step 2
   - `GEMINI_API_KEY` -> the key from step 1
4. Click **Apply** / **Create Web Service**. The build installs Python deps,
   builds the React app, and copies it into `backend/static` so Flask serves
   everything from one URL.
5. Once it's live, Render gives you a URL like
   `https://clinical-qa-assistant.onrender.com` — that's your deployed app.

## Local development (optional, before deploying)

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
