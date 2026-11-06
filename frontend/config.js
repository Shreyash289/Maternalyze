// Use relative URLs in production, fallback to localhost for development
const API_BASE_URL = (window.location.origin.includes('onrender.com') || window.location.origin.includes('tanisheesh.in'))
    ? window.location.origin
    : "http://127.0.0.1:8000";