<template>
    <div class="LR">
        <h1 align=left>Linear Regression Model</h1>
        <br />
        <h2 align=left>Phase 1: File upload </h2>
        <div>
            <h3 align = left>First, please upload your dataset for training. </h3>
            <em>Please upload a csv file.</em>
            <form @submit.prevent="submitFile">
                <input type="file" name="file" />
                <button type="submit">Submit</button>
            </form>
            <br />
        </div>
        <h2 align=left>Phase 2: Data preprocessing</h2>
        <h3 align=left>This phase removes missing values from your dataset and scales your data. </h3>
        <h3 align=left>(1). Removing missing values: </h3>
        <em>The preview of your dataset after removing missing values: </em>
        <br />
        <em>Please input the number (an integer > 0) of rows you would like for your dataset preview. </em>
        <br />
        <em>Remember to click the "submit" button before clicking the "Get Preview" button.</em>
        <br />

        <form @submit.prevent="getNumber">
            <input type="number" v-model="numRows">
            <button type="submit">Submit</button>
        </form>
        <br />
        <button @click.prevent="getPreview">Get Preview</button>
        <br />
        
        <div v-if="showPreview">
            <table>
                <thead>
                    <tr>
                        <th v-for="(_,index) in rmMissingValuesResult" :key="index">
                            {{index}}
                        </th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="index in Array.from({length: numRows}, (_, i) => i)" :key="index">
                        <td v-for="(_,row) in rmMissingValuesResult" :key="row">
                            {{rmMissingValuesResult[row][index]}}
                        </td>
                    </tr>
                </tbody>
            </table>
            
        </div>

        <br />
        <h3 align=left>(2). Scaling the dataset.</h3>

        <p style="width: 80%;">Scaling is the process of transforming the values of input data to a similar scale or range. This is often done in 
            machine learning models to improve the algorithm's performance and ensure that no input feature has an undue 
            influence on the results.</p>
        <p style="width: 80%;">In many machine learning models, such as linear regression or k-nearest neighbors, the scale of input features can 
            significantly impact the results. For example, if one feature has values much larger than the other, it may dominate 
            the model and cause it to perform poorly. Scaling can help to mitigate this issue by bringing all features to a similar range.</p>
        <p style="width: 80%;">Common methods for scaling data include standardization and normalization. Standardization scales the data with a 
            mean of 0 and a standard deviation of 1, while normalization scales the data to a range between 0 and 1. 
            Other methods, such as min-max scaling or log transformation, may be used depending on the specific requirements of the model.</p>
        <p>First, please select one column for X (input) and one column for Y (output). </p>
        <em>Enter the index names of the columns (complete names including symbols): </em>
        <br />
        <em>(You can refer to the indexes above from the data preview or your original CSV file.) </em>
        <br />
        
        <div>
            <em for="xColumn">X column: </em>
            <input type="text" id="x_index" v-model="x_index"/>
            <br />
            <em for="yColumn">Y column: </em>
            <input type="text" id="y_index" v-model="y_index" />
            <br />
            <br />
            <em>Please select the scaling mode you want to use: </em>
            <br />
            <em>Normalization: Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))</em>
            <br />
            <em>Standardization: Y = (Y - np.mean(Y)) / np.std(Y)</em>
            <br />
            <em for="scale">Enter the scaling mode you want to use: </em>
            <br />
            <input type="scale" id="scaleMode" v-model="scaleMode"/><em> (Input "Normalization" or "Standardization" here.)</em>
            <br />
            <button @click="submitParams">Submit</button>
            <br />
            <em>Click the button below to get the scatter image: </em>.
            <br />
            <button @click="scaling" v-if="scatterResource ==''" >Get Scatter Image</button>

        </div>
        
        <h3 align=left>Scatter chart of your dataset: </h3>
        <img :src="`data:image/png;base64,${scatterResource}`" v-if="scatterResource!=''"/>
        <br />

        <h2 align=left>Phase 3: Data visualization</h2>
        <p style="width: 80%;">This phase uses the train_test_split function to split your dataset into training and testing datasets. 
            It is a method provided by the scikit-learn library in Python that is commonly used for splitting a dataset into training 
            and testing subsets. This function takes in one or more arrays or matrices from your dataset. It splits them into random 
            train and test subsets, where the data in the training subset is used for training a machine learning model, and the data 
            in the test subset is used for evaluating the performance of the trained model.</p>
        <p style="width: 80%;">The test_size parameter in the train_test_split function is used to specify the proportion of the dataset 
            that should be allocated to the test set. It takes a float value between 0 and 1, and represents the fraction of the dataset 
            that should be assigned to the test subset. For example, if test_size=0.2, 20% of the data will be used for testing and 80% 
            for training.</p>
        <p style="width: 80%;">The random_state parameter is used to control the randomness of the data-splitting process. It is an integer 
            value that is used to seed the random number generator used by the train_test_split function. By setting the random_state 
            parameter to a fixed value, we can ensure that the same random train-test split is generated every time the code is executed, 
            which makes our results reproducible.</p>
        <em>Please select the parameters you want to use: </em>
        <form @submit.prevent="dataPreprocess">   
            <em for="testSize">test_size = </em>
            <input type="text" id="testSize" v-model="test_size" pattern="^[1-9][0-9]?$" />
            <em>% (Input percentage here.)</em>
            <br />
            <em for="randomState">random_state = </em>
            <input type="text" id="randomState" v-model="random_state" pattern="^[0-9]*$" />
            <br />
            <input type="submit" value="Submit" />
        </form>

        <h3 align=left>Scatter charts of your train and test datasets: </h3>
        <img :src="`data:image/png;base64,${trainTestResource}`" v-if="trainTestResource !=='' "/>
    

        <h2 align=left>Phase 4: Model training</h2>
        <h3 align=left>This phase trains your dataset in the backend and returns the predicted results here. </h3>
        <em>The prediction chart from your dataset: </em>
        <button @click="getPredict" v-if="predictionImg =='' ">Get Prediction</button>
        <br />
        <img :src="`data:image/png;base64,${predictionImg}`" v-if="predictionImg !=='' " />

        <h2 align=left>Phase 5: Accuracy</h2>
        <h3 align=left>This phase shows some accuracy results.</h3>
        <p style="width: 80%;">The Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) are commonly used 
            metrics for evaluating the performance of a regression model. The Mean Absolute Error (MAE) is the average absolute difference 
            between predicted and actual values.</p>
        <p style="width: 80%;">The Mean Squared Error (MSE) is the average squared difference between predicted and actual values. 
            The MSE measures the average magnitude of the squared errors in the predictions. Because it is squared, the MSE gives more 
            weight to larger errors, making it more useful than MAE in some cases.</p>
        <p style="width: 80%;">The Root Mean Squared Error (RMSE) is the square root of the MSE. The RMSE measures the standard deviation of 
            the errors in the predictions. It is typically used when we want to penalize larger errors more heavily than smaller ones and also 
            want to report the error in the same units as the target variable.</p>
        <em>The calculated errors from your dataset: </em>
        <button @click="getCalculation">Get Accuracy</button>
        <div v-if="showAccuracy">
            <div v-for="(value, index) in showAccuracy" v-bind:key="index">
                {{index}} {{value}}
            </div>
        </div>
        <br />
            
    </div>
</template>

<script>
    import {
        defineComponent
    } from 'vue'
    import axios from "axios"
    export default defineComponent({
        name: 'LRView',
        
        methods: {
            async submitFile(event) {
                console.log(event.target[0].files[0])
                const formData = new FormData()
                formData.append('file', event.target[0].files[0])
                try {
                    const response = await axios.post('http://localhost:5001/datasets/linear_regression', formData, {
                        headers: {
                            'Content-Type': 'multipart/form-data'
                        }
                    })
                    alert("CSV file uploaded successfully!")
                    localStorage.setItem('id', response.data.id)
                } catch (e) {
                    console.log(e)
                }
            },
            getNumber() {
                alert("Number of rows get!")
                console.log(this.number);
            },
            async getPreview(){
                try{
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem('id')}/linear_regression/missing_values`)
                    this.rmMissingValuesResult = res.data
                    this.showPreview = true
                }catch(e){
                    console.log(e)
                }
            },
            async submitParams() {
                try {
                    const data = {
                        'x_index': this.x_index,
                        'y_index': this.y_index,
                        'scaleMode': this.scaleMode
                    };
                    const response = await axios.post(`http://localhost:5001/datasets/${localStorage.getItem('id')}/linear_regression/get_columns`, data, {
                        headers: {
                            'Content-Type': 'application/json',
                        },
                    });
                    alert("X and Y columns and scale mode selected successfully!");
                    localStorage.setItem(response.data);
                } catch (e) {
                    console.log(e);
                }
            },
            async scaling(){
                try {
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem('id')}/linear_regression/scatter`, {
                        params: {
                            'x_index': this.x_index,
                            'y_index': this.y_index,
                            'scaleMode': this.scaleMode
                        }
                    });
                    console.log(res.data);
                    this.scatterResource = res.data.imgScatter;
                } catch (e) {
                    console.log(e);
                }
            },
            async dataPreprocess(event){
                let params={}
                if(event.target[0].value && event.target[0].value != ""){
                    params.test_size=parseInt(event.target[0].value)*0.01
                }
                if(event.target[1].value && event.target[1].value != ""){
                    params.random_state=parseInt(event.target[1].value)
                }
                try {
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem("id")}/linear_regression/train_test_datasets`, {params});
                    console.log(res.data);
                    event.trainTestResource = res.data.trainTestImg;
                } catch (e) {
                    console.log(e);
                }
            },
            async getPredict(){
                try{
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem("id")}/linear_regression/model_training_result`)
                    console.log(res.data)
                    this.predictionImg = res.data.imgPrediction
                } catch (e) {
                    console.log(e)
                }
            },
            async getCalculation(){
                try{
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem("id")}/linear_regression/calculation`)
                    
                    this.showAccuracy = res.data
                    this.showAccuracyValue = true
                }catch(e){
                    console.log(e)
                }
            },
            
        },
        data() {
            return {
                file: "",
                numRows: 0,
                x_index: "",
                y_index: "",
                scaleMode: "",
                test_size: "",
                random_state: "",
                showPreview: false,
               
                rmMissingValuesResult: null,
                scatterResource: "",
                trainTestResource: "",
                predictionImg: "",
                showAccuracy: {},
                showAccuracyValue: false
            }
        },
    })
</script>