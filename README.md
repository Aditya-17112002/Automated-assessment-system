# An Efficient System for Automated Assessment Evaluation ✍️📊

This project automates the evaluation of subjective handwritten answers using OCR (Optical Character Recognition) and NLP (Natural Language Processing) techniques. It is designed to reduce manual effort, improve evaluation consistency, and provide instant AI-generated feedback.

---

## 🔍 Features

- 📝 OCR-based handwritten text recognition (using Google Cloud Vision API)
- 🧠 NLP-based semantic answer evaluation using TF-IDF + cosine similarity
- 🤖 AI-generated feedback using Gemini 1.5 Turbo API
- 📈 Automated scoring based on keyword and semantic relevance
- 🧾 Clean UI with Django + HTML/CSS/JS

---

## ⚙️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Django (Python)
- **OCR**: Google Cloud Vision API
- **NLP**: NLTK, Scikit-learn (TF-IDF + Cosine Similarity)
- **AI Feedback**: Google Gemini 1.5 Turbo API
- **Deployment**: Render / GitHub

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Aditya-17112002/Automated-assessment-system.git
cd Automated-assessment-system
```

### 2. Create Virtual Environment

```bash
python -m venv env
env\Scripts\activate   # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up `.env` File

Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
```

> ⚠️ Do **not** upload your `.env` or `.json` credentials to GitHub.

---

## 🧪 How It Works

1. User uploads an image of a handwritten answer
2. Google Cloud Vision extracts text (OCR)
3. Preprocessing and TF-IDF vectorization are applied
4. Cosine similarity is computed against model answer
5. Gemini API generates feedback
6. Final score and suggestions are displayed

---


## 🧑‍💻 Author

**Aditya Arun Randive**  
GitHub: [@Aditya-17112002](https://github.com/Aditya-17112002)

---

## 📜 License

This project is for academic and learning purposes only.