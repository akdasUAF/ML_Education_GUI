# ML_Education_GUI
GUI to teach ML model to beginners

Frontend: NodeJS (Framework: vue.js)  
Backend: Python (Framework: Flask) 

## Frontend Configuration
### User Interface (UI)
Main page: Contains links to one designated CSV file and six models.
Subpages: Each subpage contains the flow process of the corresponding model, starting from uploading the CSV file, to getting the accuracy evaluation results.

### API Design
All APIs are in ``mlt-backend/server.py``. Each model has its own set of APIs. Modifying the request link of the API requires modifying the asynchronous method in the vue file of the corresponding model in ``mlt-frontend`` simultaneously.



