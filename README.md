# DataMasters-TUT21-
## Student Learning Platform

### Overview

Initially The main objective was to design an application for student living at Vaal University Housing so due to shortage of time we decided to a web application on student learning platform.

The Student Learning Platform is a web-based tool designed to help students improve their understanding of subjects such as Algebra, Geometry, and Probability. This platform provides quizzes, real-time feedback, performance tracking, and personalized study recommendations. It also uses a machine learning model to predict future quiz performance based on student history, making it an adaptive and data-driven learning experience.
This project is a collaborative effort to enhance students' learning by combining modern web technologies and machine learning algorithms.

Features

•	User Registration & Login: Students can create an account, log in, and access personalized quizzes.
•	Quizzes by Category: Take quizzes in specific subjects like Algebra, Geometry, and Probability.
•	Real-Time Feedback: Get immediate feedback on quiz performance, including correct and incorrect answers.
•	Resource Recommendations: Receive links to study resources tailored to the student's needs based on their quiz performance.
•	Performance Prediction: A Linear Regression model predicts future quiz scores based on previous attempts.
•	Softbot: A virtual assistant that guides students through quizzes, provides feedback, and offers study suggestions.

Technology Stack

•	Frontend:
o	HTML, CSS for the user interface.
•	Backend:
o	Bottle (Python) for routing and server-side logic.
o	JSON for data storage (quiz questions, user data).
•	Machine Learning:
o	scikit-learn for the Linear Regression model to predict student performance.
o	Pandas for data manipulation and handling quiz results.
o	Matplotlib for visualizing student performance trends.
•	Softbot: Integrated to offer personalized guidance and real-time feedback.
Machine Learning Integration
We implemented a Linear Regression model that predicts students' future quiz performance based on their historical data, such as:
•	Number of quiz attempts.
•	Correct and incorrect answers.
•	Scores by category (e.g., Algebra, Geometry, Probability).

This feature helps students understand their learning trajectory and provides insights into areas where they can improve.
