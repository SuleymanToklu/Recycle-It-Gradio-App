Recycle-It! ♻️ - Waste Classification Application
About the Project
Recycle-It! is a machine learning-based web application developed to help with the correct separation of waste. Users can quickly learn which category a piece of waste belongs to (cardboard, glass, metal, paper, plastic, trash) by uploading a photo or taking a picture with their camera. The model is a Convolutional Neural Network trained with the MobileNetV2 architecture.
Features
Modern and user-friendly Gradio interface
Support for both file upload and real-time photo capture via camera
Classification for 6 different waste types: cardboard, glass, metal, paper, plastic, trash
Quick testing with sample images
Informative main page (About tab)
Installation and Usage
Install the required libraries:
Apply to train_pytorc...
Run
Start the application:
Apply to train_pytorc...
Run
Go to http://127.0.0.1:7860 in your browser.
What Has Been Done in the Project
Loading a PyTorch-trained model and prediction function
Gradio interface for uploading images or taking photos via camera
Fixed incorrect sample paths, now examples are shown in the interface
Added a main page (About tab) with project and usage information
Fixed camera prediction, no error if no image is uploaded
Simplified Gradio interface, removed unnecessary parameters
Cleaned up unnecessary notebook and model files
All changes committed to GitHub
Future Improvements / Ideas
Model Improvement: Increase accuracy with larger datasets or different architectures
Mobile Optimization: Make the interface more mobile-friendly
Multi-language Support: Add support for English and other languages
User Feedback: Allow users to rate predictions and improve the model over time
API Support: Add REST API or other integrations for external applications
Data Upload & Labeling: Allow users to upload and label new data for continuous model updates
More Waste Types: Add more classes for more detailed classification
Contributing
If you want to contribute, please fork the project and send a pull request or open an issue.
Feel free to ask if you want to add or change anything!
