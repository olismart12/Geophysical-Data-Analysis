import os

# List of libraries to install
libraries = [
    'pandas',
    'numpy',
    'scikit-learn'
]

# Function to install libraries using pip
def install_libraries(libs):
    for lib in libs:
        os.system(f"pip install {lib}")

if __name__ == "__main__":
    install_libraries(libraries)
    print("Libraries installed successfully.")
