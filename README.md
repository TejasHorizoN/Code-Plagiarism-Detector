# 🧬 TwinCode: Find Your Code's Twin

**TwinCode** is a modern, web-based code similarity detection tool. With an immersive UI and advanced fingerprinting algorithms, TwinCode helps you instantly analyze, visualize, and report code similarity—making it easy to ensure code authenticity and academic integrity.

[![View on GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&style=flat-square)](https://github.com/TejasHorizoN/Code-Plagiarism-Detector)

---

## ✨ Features

- **Modern, Immersive UI**  
  - Gradient backgrounds, glassmorphism cards, animated elements, and a clean, tab-based layout.
- **Flexible File Support**  
  - Upload and analyze `.py`, `.c`, `.cpp`, `.java`, `.js`, `.ts`, `.php`, `.rb`, `.go`, `.rs`, `.txt` and more.
  - All files in a batch must have the same extension (auto-validated).
- **One-Click Detection**  
  - Upload two or more files, click "Run Plagiarism Detection," and get instant results.
- **Interactive Analytics Dashboard**  
  - **Highest Similarity Metric:** Instantly see the most critical similarity score.
  - **Risk Level Assessment:** Categorizes the highest similarity as "Low," "Medium," or "High" risk.
  - **Similarity Distribution Histogram:** Visualize the spread of similarity scores.
  - **Similarity Matrix Heatmap:** See a grid of all file-to-file similarity scores.
  - **Detailed Results Table:** Downloadable CSV of all pairwise results.
- **Comprehensive HTML Reports**  
  - Download a full HTML report with side-by-side code and highlighted matches.
- **Analysis History & Trends**  
  - Track all your detection jobs, with a trend chart of maximum similarity over time.
- **Help & Documentation**  
  - Step-by-step guide, FAQ, and algorithm explanation built in.

---

## 🚦 How It Works

TwinCode uses advanced fingerprinting technology:
- **Tokenization:** Breaks code into tokens.
- **K-gram Analysis:** Creates overlapping token sequences.
- **Winnowing Algorithm:** Selects representative fingerprints.
- **Similarity Scoring:** Calculates percentage overlap between files.

**Supported Formats:**  
Common source code files (`.py`, `.c`, `.cpp`, `.java`, `.js`, `.ts`, `.php`, `.rb`, `.go`, `.rs`, `.txt`, etc.)

**Performance:**  
Optimized for fast analysis of large codebases with efficient memory usage.

---

## 🛠️ Usage Workflow

1. **Upload Files:**  
   Upload 2 or more code files using the file uploader. All files must have the same extension.

2. **Run Detection:**  
   Click the **Run Plagiarism Detection** button to start the analysis.

3. **Review Results:**  
   - View metrics, analytics, and risk levels.
   - Download detailed CSV and HTML reports.
   - Explore your analysis history and trends.

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TejasHorizoN/Code-Plagiarism-Detector.git
   cd Code-Plagiarism-Detector
   ```

2. **Install dependencies:**  
   Make sure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   The app will open in your browser!

---

## 📄 Example Use Case

- Upload two or more `.py` or `.c` files.
- Click **Run Plagiarism Detection**.
- Instantly view analytics, similarity scores, and download a detailed HTML report.

---

## 🧑‍💻 Credits

Made with ❤️ by Team Fanatic  
[View on GitHub](https://github.com/TejasHorizoN/Code-Plagiarism-Detector)

---
