<template>
    <div class="Neural">
        <h1 align=left>Neural Network Model</h1>
        <h3>For the logistic regression, SVM, k-means clustering, and neural network models, please use the CSV file from the above link: </h3>
          <a href="http://nrvis.com/data/mldata/breast-cancer-wisconsin_wdbc.csv" target="_blank">
            <h4>breast-cancer-wisconsin_wdbc.csv (click on the link will directly download the CSV file)</h4>
          </a>
        <br />
        <h2 align=left>Phase 1: File upload </h2>
        <div>
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
        <button @click="getPreview">Get Preview</button>
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
        <h3 align=left>(2). Scaling the dataset: </h3>
        <p style="width: 80%;">Scaling is the process of transforming the values of input data to a similar scale or range. This is often done in 
            machine learning models to improve the algorithm's performance and ensure that no input feature has an undue 
            influence on the results.</p>
        <p style="width: 80%;">In many machine learning models, such as linear regression or k-nearest neighbors, the scale of input features can 
            significantly impact the results. For example, if one feature has values much larger than the other, it may dominate 
            the model and cause it to perform poorly. Scaling can help to mitigate this issue by bringing all features to a similar range.</p>
        <p style="width: 80%;">Common methods for scaling data include standardization and normalization. Standardization scales the data with a 
            mean of 0 and a standard deviation of 1, while normalization scales the data to a range between 0 and 1. 
            Other methods, such as min-max scaling or log transformation, may be used depending on the specific requirements of the model.</p>
        <p style="width: 80%">(In this case, we are using "breast-cancer-wisconsin_wdbc.csv", X will contain all columns from the third column (index 2) to the last column in this CSV file; 
            y will contain the values from the second column (index 1) of this CSV file.)</p>
        <em>Please select the scaling mode you want to use: </em>
        <br />
        <em>Normalization: Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))</em>
        <br />
        <em>Standardization: Y = (Y - np.mean(Y)) / np.std(Y)</em>
        <br />
        <form @submit.prevent="scaleMode">
            <select name="scaling" id="scale">
            <option value="normalization">Normalization</option>
            <option value="standardization">Standardization</option>
            </select>
            <input type="submit" value="Submit" />
            
        </form>
        
        <h3 align=left>Feature labels of your dataset: </h3>
        <div v-if="features">
            <div v-for="i in features" v-bind:key="i">
                {{i}} {{features[i]}}
            </div>
        </div>
        
        <h3 align=left>Phase 3: Data visualization</h3>
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
        <br />
        <em>Please select the parameters you want to use: </em>
        <form @submit.prevent="dataPreprocess">
            
        <em for="testSize">test_size = </em>
        <input type="text" id="testSize" pattern="^[1-9][0-9]?$" />
        <em>% (Input percentage here.)</em>
        <br />
        <input type="submit" value="Submit" />
        </form>

        <h3 align=left>Shapes of the split datasets: </h3>
        <div v-if="shapes">
            <div v-for="(value, index) in shapes" v-bind:key="index">
                {{index}} {{value}}
            </div>
        </div>

        <h3 align=left>Phase 4: Model training</h3>
        <em>The prediction chart from your dataset: </em>
        <button @click="getPredict" v-if="trainTestResource =='' ">Get Prediction</button>
        <br />
        <img :src="`data:image/png;base64,${trainTestResource}`" v-if="trainTestResource !=='' " />

        <h3 align=left>Phase 5: Accuracy</h3>
        <p style="width: 80%;">The Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) are commonly used 
            metrics for evaluating the performance of a regression model. The Mean Absolute Error (MAE) is the average absolute difference 
            between predicted and actual values.</p>

        <p style="width: 80%;">The Mean Squared Error (MSE) is the average squared difference between predicted and actual values. 
            The MSE measures the average magnitude of the squared errors in the predictions. Because it is squared, the MSE gives more 
            weight to larger errors, making it more useful than MAE in some cases.</p>

        <p style="width: 80%;">The Root Mean Squared Error (RMSE) is the square root of the MSE. The RMSE measures the standard deviation of 
            the errors in the predictions. It is typically used when we want to penalize larger errors more heavily than smaller ones and also 
            want to report the error in the same units as the target variable.</p>

        <em>The calculated errors from your created model: </em>
        <br />
        <button @click="getCalculation">Get Model Accuracy</button>
        <div v-if="showAccuracy">
            <div v-for="(value, index) in showAccuracy" v-bind:key="index">
                {{index}} {{value}}
            </div>
        </div>
        <br />
        <button @click="getTrainErrors">Get Errors on Train Data</button>
        <div v-if="showTrainErrors">
            <div v-for="(value, index) in showTrainErrors" v-bind:key="index">
                {{index}} {{value}}
            </div>
        </div>
        <br />
        <button @click="getTestErrors">Get Errors on Test Data</button>
        <div v-if="showTestErrors">
            <div v-for="(value, index) in showTestErrors" v-bind:key="index">
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
    axios.defaults.baseURL = process.env.VUE_APP_BACKEND_URL
    export default defineComponent({
        name: 'NeuralView',
        methods: {
            async submitFile(event) {
                console.log(event.target[0].files[0])
                const formData = new FormData()
                formData.append('file', event.target[0].files[0])
                try {
                    const response = await axios.post('/datasets/neural_network', formData, {
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
                    const res = await axios.get(`/datasets/${localStorage.getItem('id')}/neural_network/missing_values`)
                    this.rmMissingValuesResult = res.data
                    this.showPreview = true
                }catch(e){
                    console.log(e)
                }
            },
            async scaleMode(event){
                try{
                    const res = await axios.get(`/datasets/${localStorage.getItem("id")}/neural_network/get_features?scaleMode=${event.target[0].value}`)
                    this.features = res.data
                    this.featuresValue = true
                    
                } catch (e) {
                    console.log(e)
                }
            },
            async dataPreprocess(event){
                let params={}
                if(event.target[0].value && event.target[0].value != ""){
                    params.test_size=parseInt(event.target[0].value)*0.01
                }
                try{
                    const res=await axios.get(`/datasets/${localStorage.getItem("id")}/neural_network/datasets_shapes`,{params})
                    this.shapes = res.data
                    this.shapesValue = true
                    
                } catch (e) {
                    console.log(e)
                }
            },
            async getPredict(){
                try{
                    const res = await axios.get(`/datasets/${localStorage.getItem("id")}/neural_network/model_training_result`)
                    console.log(res.data)
                    this.trainTestResource = res.data.confsMatrix
                } catch (e) {
                    console.log(e)
                }
            },
            async getCalculation(){
                try{
                    const res = await axios.get(`/datasets/${localStorage.getItem('id')}/neural_network/calculation`)
                    
                    this.showAccuracy = res.data
                    this.showAccuracyValue = true
                }catch(e){
                    console.log(e)
                }
            },
            async getTrainErrors(){
                try{
                    const res = await axios.get(`/datasets/${localStorage.getItem('id')}/neural_network/get_train_data_errors`)
                    
                    this.showTrainErrors = res.data
                    this.showTrainErrorsValue = true
                }catch(e){
                    console.log(e)
                }
            },
            async getTestErrors(){
                try{
                    const res = await axios.get(`/datasets/${localStorage.getItem('id')}/neural_network/get_test_data_errors`)
                    
                    this.showTestErrors = res.data
                    this.showTestErrorsValue = true
                }catch(e){
                    console.log(e)
                }
            },
            
        },
        data() {
            return {
                file: "",
                numRows: 0,
                showPreview: false,
                features: {},
                featuresValue: false,
                rmMissingValuesResult: null,
                shapes: {},
                shapesValue: false,
                
                trainTestResource: "",
                showAccuracy: {},
                showAccuracyValue: false,
                showTrainErrors: {},
                showTrainErrorsValue: false,
                showTestErrors: {},
                showTestErrorsValue: false,
            }
        },
    })
</script>