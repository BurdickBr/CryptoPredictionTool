import './App.css';
import React, { useState, Component, ComponentProps } from "react";
import Select from 'react-select'
import {Button} from 'react-bootstrap'
import { PieChart } from 'react-minimal-pie-chart';

const props = {
  data: ComponentProps<typeof PieChart>['data']
}

function App(props) {

  // States for select
  const [selectState, setSelectState] = useState('BTC')

  // States  For pie chart
  const [probIncrease, setProbIncrease] = useState(50)
  const [probDecrease, setProbDecrease] = useState(50)
  const [increaseColor, setIncreaseColor] = useState('darkgreen')
  const [decreaseColor, setDecreaseColor] = useState('maroon')
  const [selected, setSelected] = useState(0);
  const [hovered, setHovered] = useState(undefined);

  async function getPrediction() {
    console.log('getting predictions...')  

    let jsondata;    
    await fetch('http://127.0.0.1:8000/prediction_generator/' + selectState).then(
            function(u){ return u.json();}
          ).then(
            function(json){
              jsondata = json;
            }
          )

    var json = JSON.parse(jsondata)
    
    setProbIncrease(json['1'] * 100)
    setProbDecrease(json['2'] * 100)
  }

  function handleSelectChange(e) {
    setSelectState(e.target.value)
    console.log('selection made')
  }


  const lineWidth = 50

  return (
    <div className="App">
      <h1>Crypto Price Predictor!</h1>
      <div className="MyButton">
        <Button onClick={getPrediction}>Get Results</Button>
      </div>
      <div>
        <select
          className="MySelect"
          value={selectState} 
          onChange={handleSelectChange}
        >
          <option value='BTC'>BTC</option>
          <option value='ETH'>ETH</option>
          <option value='AVAX'>AVAX</option>
          <option value='AAVE'>AAVE</option>
          <option value='BCH'>BCH</option>
        </select>
      </div>  
      <div className="MyPieChart">
        <PieChart
        style={{
          fontFamily:
            '"Nunito Sans", -apple-system, Helvetica, Arial, sans-serif',
          fontSize: '8px',
        }}
        data={[
          { title: "ProbIncrease", value:probIncrease, color: increaseColor, key: 'increase'},
          { title: "ProbDecrease", value:probDecrease, color: decreaseColor, key: 'decrease'}
        ]}
        radius={PieChart.defaultProps.radius - 6}
        lineWidth={60}
        segmentsStyle={{ transition: 'stroke .3s', cursor: 'pointer' }}
        segmentsShift={(index) => (index === selected ? 6 : 1)}
        animate
        label={({ dataEntry }) => Math.round(dataEntry.percentage) + '%'}
        labelPosition={100 - lineWidth / 2}
        labelStyle={{
        fill: '#fff',
        opacity: 0.75,
        pointerEvents: 'none',
        }}
        onClick={(_, index) => {
          setSelected(index === selected ? undefined : index);
        }}
        onMouseOver={(_, index) => {
          setHovered(index);
          if (index===0) {
            setIncreaseColor('grey')
          }
          if (index===1) {
            setDecreaseColor('grey')
          }

        }}
        onMouseOut={() => {
          setHovered(undefined);
          setIncreaseColor('maroon')
          setDecreaseColor('darkgreen')
        }}


        ></PieChart>
      </div>
    </div>
  );
}

export default App;