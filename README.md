# Advanced Deepfake Detector

An interactive web application for detecting deepfake images using deep learning, built with Streamlit.

## Features

- 🔍 **Advanced Detection Mode**: Upload and analyze images for deepfake detection
- 🎮 **Challenge Mode**: Test your skills against the AI in identifying deepfakes
- 📚 **Learn Mode**: Educational resources about deepfake detection
- 📊 **Statistics**: Track your performance and export results
- 🎯 **Achievements**: Earn achievements as you improve
- 🎨 **Customizable**: Light/Dark theme support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Local Development

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Deployment

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t deepfake-detector .
```

2. Run the container:
```bash
docker run -d -p 8501:8501 --name deepfake-detector deepfake-detector
```

### Cloud Deployment

#### Streamlit Cloud
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Deploy directly from your GitHub repository

#### Heroku
1. Install Heroku CLI
2. Login to Heroku:
```bash
heroku login
```

3. Create a new Heroku app:
```bash
heroku create your-app-name
```

4. Deploy:
```bash
git push heroku main
```

### Security Considerations

1. Model Security:
   - Ensure model files are properly secured
   - Implement rate limiting
   - Validate all inputs

2. Data Privacy:
   - No user data is stored permanently
   - Images are processed in memory
   - Statistics are stored locally

3. System Requirements:
   - CPU: 2+ cores recommended
   - RAM: 4GB minimum
   - Storage: 1GB for model and application

## Monitoring

The application includes built-in logging and monitoring:
- Access logs in `.streamlit/logs`
- Performance metrics in the admin panel
- Error tracking and reporting

## Troubleshooting

Common issues and solutions:
1. Model loading errors:
   - Verify model files are in the correct location
   - Check file permissions
   
2. Memory issues:
   - Increase available RAM
   - Monitor system resources

3. Image processing errors:
   - Check supported image formats
   - Verify file size limits

## Modes

### Detector Mode
- Upload your own images
- Get detailed analysis with confidence scores
- View heatmaps of suspicious areas
- Track analysis history

### Challenge Mode
- Test your ability to spot deepfakes
- Compete against the AI
- Track your score and achievements
- Improve your detection skills

### Learn Mode
- Understanding deepfakes
- Common detection techniques
- Real-world case studies
- Additional resources

## Requirements

- Python 3.8+
- See requirements.txt for full list of dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 