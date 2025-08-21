# 💡 ThornLumina Seeker

**AI-Powered Lighting Product Discovery Engine**

Transform your product catalogs into intelligent, searchable databases with natural language processing capabilities.

![ThornLumina Seeker](https://img.shields.io/badge/AI-Powered-brightgreen) ![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red) ![Python](https://img.shields.io/badge/Python-3.8+-blue)

## 🌟 Features

- **📄 PDF Catalog Processing**: Upload lighting catalogs, datasheets, and specifications
- **🔍 Natural Language Search**: Query products using everyday language
- **⚡ Instant Results**: Fast AI-powered product matching
- **📊 Structured Output**: Clean JSON format with detailed specifications
- **🎯 Smart Recommendations**: AI understands context and intent

## 🚀 Live Demo

**Streamlit Cloud**: [Your deployed app URL will go here]

## 📋 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/thornlumina-seeker.git
   cd thornlumina-seeker
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account
2. **Connect to Streamlit Cloud** at [share.streamlit.io](https://share.streamlit.io)
3. **Deploy** by selecting your forked repository
4. **Share** your live app with the world!

## 🎯 How It Works

### 1. Upload Catalog
Upload your PDF lighting catalog or product specification document.

### 2. AI Processing
Our AI engine:
- Extracts text from your PDF
- Splits content into searchable chunks
- Creates vector embeddings for semantic search
- Builds an intelligent search index

### 3. Natural Language Search
Search using natural language queries like:
- "20W LED bulb for office lighting"
- "Outdoor floodlight with 2000 lumens"
- "Energy efficient lamp for residential use"
- "High CRI LED strip for photography"

### 4. Structured Results
Get detailed product information in clean JSON format:
```json
{
  "product_name": "ThornLumina Pro LED 20W",
  "application": "Office and commercial lighting",
  "wattage": "20W",
  "luminous_flux": "2000 lumens",
  "voltage": "220-240V",
  "additional_features": "Dimmable, 4000K, IP54 rated"
}
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: Hugging Face Transformers, LangChain
- **Search**: FAISS Vector Database
- **NLP**: Sentence Transformers
- **PDF Processing**: PyPDF
- **Language Model**: Google FLAN-T5

## 📁 Project Structure

```
thornlumina-seeker/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore rules
├── .streamlit/          # Streamlit configuration
│   └── config.toml      # App settings
└── assets/              # Static assets (optional)
    └── demo_catalog.pdf # Sample catalog for testing
```

## 🎨 Use Cases

### Lighting Manufacturers
- Product catalog digitization
- Customer self-service portals
- Sales team product lookup tools

### Electrical Distributors
- Inventory search systems
- Quote generation assistance
- Product recommendation engines

### Architects & Engineers
- Specification research
- Product comparison tools
- Project planning assistance

## 🐛 Troubleshooting

### Common Issues

**Issue**: Model download fails
```bash
# Solution: Pre-download models
python -c "from transformers import pipeline; pipeline('text2text-generation', model='google/flan-t5-base')"
```

**Issue**: Memory errors on large PDFs
- Reduce chunk_size to 500 in the text splitter
- Use smaller PDF files for testing

**Issue**: Slow initial startup
- First run downloads models (~1GB)
- Subsequent runs are much faster

## 📈 Roadmap

- [ ] Support for multiple file formats (Word, Excel)
- [ ] Advanced filtering options
- [ ] Product comparison features
- [ ] Export search results
- [ ] Custom model fine-tuning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Hugging Face** for transformer models
- **LangChain** for document processing
- **Streamlit** for the beautiful web framework
- **FAISS** for efficient vector search

---

**Made with ❤️ for the lighting industry**
