import './App.css';
import React, { useState, Component } from "react";
import Select from 'react-select'
import {Button} from 'react-bootstrap'
import { PieChart } from 'react-minimal-pie-chart';

function testGet() {
  console.log('getting test')
  fetch('http://127.0.0.1:8000/prediction_generator/ETH')
    .then(response => response.json())
    .then(data => console.log(data));
}

const options = [
  { value: 'BTC', label: 'BTC' },
  { value: 'ETH', label: 'ETH' },
  { value: 'AVAX', label: 'AVAX' }
]

const pie_data = [
  { title: 'Probability of Increase', value: 50, color: "#E38627"},
  { title: 'Probability of Decrease', value: 50, color: "#6A2135"}
]

const MySelect = () => (
  <Select options={options} />
)

const MyPie = () => (
  <PieChart data={pie_data} />
)

function App() {
  return (
    <div className="App">
      <h1>hello world</h1>
      <div className="MyButton">
        <Button onClick={testGet}>Get Results</Button>
      </div>
      <div className="MyPieChart">
        <MyPie></MyPie>
      </div>
      <div className="MySelect">
        <MySelect></MySelect>
      </div>
      
      
      
    </div>
  );
}

export default App;