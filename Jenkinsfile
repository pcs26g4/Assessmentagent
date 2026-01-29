pipeline {
    agent any

    stages {

        stage('Check Python') {
            steps {
                bat 'python --version'
                bat 'python -m pip --version'
            }
        }

        stage('Install Dependencies') {
            steps {
                dir('server') {
                    bat 'python -m pip install --upgrade pip'
                    bat 'python -m pip install -r requirements.txt'
                }
            }
        }

        stage('Build (Smoke Test)') {
            steps {
                dir('server') {
                    bat 'python -m py_compile main.py'
                }
            }
        }
        
        stage('Deploy to Render') {
            steps {
                bat 'curl -X POST "%RENDER_DEPLOY%"'                
            }
        }
    }
}
