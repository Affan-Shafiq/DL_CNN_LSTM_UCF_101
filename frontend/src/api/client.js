import axios from 'axios';

// Backend API URL - Update this when deploying to Hugging Face Spaces
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const predictAction = async (videoFile) => {
    const formData = new FormData();
    formData.append('file', videoFile);

    const response = await apiClient.post('/predict', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};

export default apiClient;
