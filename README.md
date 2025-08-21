# 💡 Product Finder AI (Streamlit + LangChain + HuggingFace)

A free AI-powered product finder that:
- Uploads a product catalog PDF
- Creates a vector store with FAISS
- Lets you query products by application, wattage, luminous flux
- Returns structured product details

## 🚀 Deploy on Streamlit Cloud (Free)
1. Fork this repo to your GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Connect your GitHub → pick this repo → deploy.
4. Done! 🎉

## 🛠 Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧪 Run on Colab
Use ngrok to expose Streamlit:
```python
!pip install streamlit pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
!streamlit run app.py --server.port 8501 &
print(ngrok.connect(8501))
```
