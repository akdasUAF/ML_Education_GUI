<template>
    <div class="SVM">
        <h1 align=left>SVM Model</h1>
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
        <p style="width: 80%">(In this case, we are using "breast-cancer-wisconsin_wdbc.csv", columns 3 and 4 (indexed by idx1 and idx2) are used as features for X.
            Column 2 is used as the target variable for y.)</p>

        <em>Please select the scaling mode you want to use: </em>
        <form @submit.prevent="scaleMode">
            <select name="scaling" id="scale">
            <option value="normalization">Normalization</option>
            <option value="standardization">Standardization</option>
            </select>
            <input type="submit" value="Submit" />
            
        </form>
        
        <h3 align=left>Scatter chart of your dataset: </h3>
        <img :src="`data:image/png;base64,${scatterResource}`" v-if="scatterResource!=''"/>
    
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
        <em>Please select the parameters you want to use: </em>
        <form @submit.prevent="dataPreprocess">
            
        <em for="testSize">test_size = </em>
        <input type="text" id="testSize" pattern="^[1-9][0-9]?$" />
        <em>% (Input percentage here.)</em>
        <br />
            
        <input type="submit" value="Submit" />
        </form>

        <h3 align=left>Scatter charts of your train and test datasets: </h3>
        <img :src="`data:image/png;base64,${trainTestResource}`" v-if="trainTestResource !=='' "/>
    

        <h3 align=left>Phase 4: Model training</h3>
        <em>The prediction chart from your dataset: </em>
        <br />
        <button @click="getSolution" v-if="solutionResource =='' ">Get Solution Plot</button>
        <br />
        <img :src="`data:image/png;base64,${solutionResource}`" v-if="solutionResource !=='' " />
        <br />
        <button @click="getConfMatrix" v-if="confMatrixResource =='' ">Get Confusion Matrix Plot</button>
        <br />
        <img :src="`data:image/png;base64,${confMatrixResource}`" v-if="confMatrixResource !=='' " />

        <h3 align=left>Phase 5: Accuracy</h3>
        <p style="width: 80%;">Model accuracy is a measure of how often a classification model makes correct predictions overall. 
            It's calculated as the ratio of the number of correct predictions to the total number of predictions made by the model. 
            <br />Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)</p>

        <p style="width: 80%;">Precision is a metric that focuses on the proportion of correctly predicted positive cases (true positives) 
            out of all instances predicted as positive (true positives + false positives).
            <br />Precision = (True Positives) / (True Positives + False Positives)</p>

        <p style="width: 80%">Recall, also known as sensitivity or true positive rate, measures the proportion of correctly predicted 
            positive cases (true positives) out of all actual positive cases (true positives + false negatives).
            <br />Recall = (True Positives) / (True Positives + False Negatives)</p>

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
        name: 'SVMView',
        methods: {
            async submitFile(event) {
                console.log(event.target[0].files[0])
                const formData = new FormData()
                formData.append('file', event.target[0].files[0])
                try {
                    const response = await axios.post('http://localhost:5001/datasets/svm', formData, {
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
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem('id')}/svm/missing_values`)
                    this.rmMissingValuesResult = res.data
                    this.showPreview = true
                }catch(e){
                    console.log(e)
                }
            },
            async scaleMode(event){
                try{
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem("id")}/svm/scatter?scaleMode=${event.target[0].value}`)
                    console.log(res.data)
                    this.scatterResource = res.data.imgScatter
                    
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
                    const res=await axios.get(`http://localhost:5001/datasets/${localStorage.getItem("id")}/svm/train_test_results`,{params})
                    console.log(res.data)
                    this.trainTestResource = res.data.trainTestImg
                    
                } catch (e) {
                    console.log(e)
                }
            },
            async getSolution(){
                try{
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem("id")}/svm/show_solution`)
                    console.log(res.data)
                    this.solutionResource = res.data.solutionImg
                } catch (e) {
                    console.log(e)
                }
            },
            async getConfMatrix(){
                try{
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem("id")}/svm/show_confusion_matrix`)
                    console.log(res.data)
                    this.confMatrixResource = res.data.confMatrix
                } catch (e) {
                    console.log(e)
                }
            },
            async getCalculation(){
                try{
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem('id')}/svm/calculation`)
                    
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
                imgScatter: "",
                showPreview: false,

                rmMissingValuesResult: null,
                scatterResource: "",
                trainTestResource: "",
                solutionResource: "",
                confMatrixResource: "",
                showAccuracy: {},
                showAccuracyValue: false
            }
        },
    })
</script>