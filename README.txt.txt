All rights reserved to Mohammad Al-Obiad.

I couldn't deploy the code because it would be too costly. Therefore, to run the Index file correctly, you need to follow these steps:
- Make sure Python version 3.11 or 3.12 is installed on your device using the following command:  
  python --version

- Ensure that the `pip` tool is installed on your device by running:  
  python -m pip --version  
  If it's not installed, install it first.

- Open PowerShell as an administrator and navigate to the download folder using the command:  
  cd $PATH$

- Create a virtual environment to install libraries independently using the following commands:  
  python -m venv .env  
  .env/Scripts/activate

- Next, install the libraries listed in the `requirements.txt` file using:  
  pip install -r requirements.txt

- To download all the necessary files to run the code, execute the Python file using:  
  python3 nlpCode.py  
  or  
  python nlpCode.py  
  This will automatically download the required files. This step may take several minutes, so please wait until it is completed.

- Once the download is complete, there is no need to run the Python file again.

- Now everything is ready. Run the following command to start the application and test the model:  
  uvicorn nlpCode:app --reload  
  This will allow you to run the `Index` file and test the model without any issues.

The model can classify Turkish news headlines into the following categories:  
yaşam, spor, sağlık, ekonomi  
This is due to the dataset on which the model was trained.

Note that the Python file Multiclass news classification using Natural Language Processing.py contains the code used to create the BERT model trained on the dataset.
Thank you for trying it out!
