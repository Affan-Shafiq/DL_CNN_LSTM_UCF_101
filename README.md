# Video Action Recognition - UCF101

A full-stack application for video action recognition using CNN + LSTM trained on the UCF101 dataset.

## Project Overview

This project consists of:
- **Backend**: FastAPI server for video inference (deployed on Hugging Face Spaces)
- **Frontend**: React application for video upload and result display (deployed on GitHub Pages)
- **Model**: CNN + LSTM architecture (MobileNetV2 + LSTM) trained on UCF101 dataset subset (3000 videos)

## Architecture

```
Frontend (React) → Backend (FastAPI) → Model (TensorFlow/Keras)
    ↓                    ↓                      ↓
GitHub Pages      HuggingFace Spaces      CNN + LSTM
```

## Project Structure

```
DL_Project_CNN_LSTM/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── model.py             # Model loading and inference
│   ├── preprocess.py        # Video preprocessing
│   ├── requirements.txt     # Python dependencies
│   └── .env.example         # Environment variable template
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── api/client.js    # API client
│   │   └── index.css        # Styles
│   ├── package.json
│   ├── vite.config.js
│   └── .env.example         # Environment variable template
└── README.md
```

## Local Development

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your trained model:
   - Copy `ucf101_cnn_lstm_subset_model.h5` to the `backend/` directory

5. Run the server:
```bash
python -m uvicorn main:app --reload
```

Backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Configure API URL:
   - Copy `.env.example` to `.env.local`
   - Set `VITE_API_URL=http://localhost:8000`

4. Run the development server:
```bash
npm run dev
```

Frontend will be available at `http://localhost:5173`

## Deployment

### Backend (Hugging Face Spaces)

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select "SDK: Gradio" or "Docker"
3. Upload the following files:
   - `main.py`
   - `model.py`
   - `preprocess.py`
   - `requirements.txt`
   - `ucf101_cnn_lstm_subset_model.h5` (your trained model)
4. Create an `app.py` that imports and runs the FastAPI app
5. Your API will be available at `https://your-username-space-name.hf.space`

### Frontend (GitHub Pages)

1. Update `.env.local` with your Hugging Face Space URL:
```
VITE_API_URL=https://your-username-space-name.hf.space
```

2. Build the project:
```bash
npm run build
```

3. Deploy to GitHub Pages:
```bash
npm run deploy
```

Alternatively, push to GitHub and enable GitHub Pages in repository settings.

## Model Details

- **Architecture**: TimeDistributed(MobileNetV2) + LSTM(128) + Dense(128) + Dense(101)
- **Input**: 16 frames per video, resized to 224x224
- **Output**: Predicted action class from 101 UCF101 categories
- **Training**: Subset of 3,000 videos (2,500 train, 500 test)
- **Framework**: TensorFlow/Keras

##  Features

- Drag-and-drop video upload
- Modern, animated UI with glassmorphism design
- Real-time prediction with confidence scores
- CORS-enabled API for cross-origin requests
- Mobile-responsive design
- Error handling and loading states

## Tech Stack

**Backend:**
- FastAPI
- TensorFlow/Keras
- OpenCV
- NumPy
- Uvicorn

**Frontend:**
- React 19
- Vite
- Axios
- Framer Motion
- Modern CSS

## Notes

- The model file (`ucf101_cnn_lstm_subset_model.h5`) is not included in this repository due to size constraints
- When the model is not present, the backend uses a mock prediction for demonstration
- Ensure CORS is properly configured when deploying to production

##  Author

Built as part of a Deep Learning project for video action recognition.

## License

This project is for educational purposes.
