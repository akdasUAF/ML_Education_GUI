<template>
    <div class="Poly">
        <h1 align=left>Polynomial Regression Model</h1>
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
        
        <h3 align=left>Scatter chart of your dataset: </h3>
        <img :src="`data:image/png;base64,${scatterResource}`" v-if="scatterResource!=''"/>

        <h3 align=left>Phase 3: Data visualization</h3>
        <em>Please select the parameters you want to use: </em>
        <form @submit.prevent="dataPreprocess">
            
        <em for="testSize">test_size = </em>
        <input type="text" id="testSize" pattern="^[1-9][0-9]?$" />
        <em>% (Input percentage here.)</em>
        <br />
        <em for="randomState">random_state = </em>
        <input type="text" id="randomState" pattern="^[0-9]*$" />
        <br />
        <input type="submit" value="Submit" />
        </form>

        <h3 align=left>Scatter charts of your train and test datasets: </h3>
        <img :src="`data:image/png;base64,${trainTestResource}`" v-if="trainTestResource !=='' "/>

        <h3 align=left>Phase 4: Model training</h3>
        <em>The prediction chart from your dataset: </em>
        <button @click="getPredict" v-if="predictionImg =='' ">Get Prediction</button>
        <br />
        <img :src="`data:image/png;base64,${predictionImg}`" v-if="predictionImg !=='' " />

        <h3 align=left>Phase 5: Accuracy</h3>
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
        name: 'PolyView',
        methods: {
            async submitFile(event) {
                console.log(event.target[0].files[0])
                const formData = new FormData()
                formData.append('file', event.target[0].files[0])
                try {
                    const response = await axios.post('http://localhost:5001/datasets/polynomial_regression', formData, {
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
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem('id')}/polynomial_regression/missing_values`)
                    this.rmMissingValuesResult = res.data
                    this.showPreview = true
                }catch(e){
                    console.log(e)
                }
            },
            async scaleMode(event){
                try{
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem("id")}/polynomial_regression/scatter?scaleMode=${event.target[0].value}`)
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
                if(event.target[1].value && event.target[1].value != ""){
                    params.random_state=parseInt(event.target[1].value)
                }
                try{
                    const res=await axios.get(`http://localhost:5001/datasets/${localStorage.getItem("id")}/polynomial_regression/train_test_results`,{params})
                    console.log(res.data)
                    this.trainTestResource = res.data.trainTestImg
                    
                } catch (e) {
                    console.log(e)
                }
            },
            async getPredict(){
                try{
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem("id")}/polynomial_regression/model_training_result`)
                    console.log(res.data)
                    this.predictionImg = res.data.imgPrediction
                } catch (e) {
                    console.log(e)
                }
            },
            async getCalculation(){
                try{
                    const res = await axios.get(`http://localhost:5001/datasets/${localStorage.getItem('id')}/polynomial_regression/calculation`)
                    
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
                predictionImg: "",
                showAccuracy: {},
                showAccuracyValue: false
            }
        },
    })
</script>