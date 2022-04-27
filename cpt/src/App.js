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

const MyComponent = () => (
  <Select options={options} />
)

function App() {
  return (
    <div className="App">
      <h1>hello world</h1>
      <div className="MyButton">
        <Button onClick={testGet}>Button</Button>
      </div>
      <PieChart
      data={[
    { title: 'One', value: 10, color: '#E38627' },
    { title: 'Two', value: 15, color: '#C13C37' },
    { title: 'Three', value: 20, color: '#6A2135' },
  ]}
/>;
      <div className="MySelect">
        <MyComponent></MyComponent>
      </div>
      
      
      
    </div>
  );
}

export default App;