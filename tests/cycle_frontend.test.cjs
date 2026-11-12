const {test}=require('node:test');
const assert=require('node:assert/strict');
const {readFileSync}=require('node:fs');
const vm=require('node:vm');
const script=readFileSync('web/cycle.js','utf8').replace(/initializeCycle\(\);\s*$/, '');

test('TaskForce distinguishes experience collection from actual policy updates',()=>{
  const context=vm.createContext({});vm.runInContext(script,context);
  const status=record=>vm.runInContext('cyclePolicyStatus('+JSON.stringify(record)+')',context);
  assert.match(status({config:{coordination:'taskforce'},coordination:{policy_updates:0}}),/warming up/);
  assert.equal(status({config:{coordination:'taskforce'},coordination:{policy_updates:3}}),'Policy learning · 3 updates');
  assert.equal(status({config:{coordination:'none'}}),'');
});

test('selecting TaskForce preserves the explicit two-step training budget',()=>{
  const fields={cycleTask:{value:'Check classifier'},cycleStage:{value:'classification'},cycleSource:{value:'/source'},
    cycleData:{value:'/data'},cycleGpu:{value:'0'},cycleAcceptance:{value:'1'},cycleTraining:{value:'2'},
    cycleCoordination:{checked:true},cyclePlans:{value:''},cycleDataset:{value:''},cycleFold:{value:''}};
  const context=vm.createContext({byId:id=>fields[id]});vm.runInContext(script,context);
  const payload=vm.runInContext("cyclePayload('run')",context);
  assert.equal(payload.coordination,'taskforce');assert.equal(payload.train_steps,2);assert.equal(payload.acceptance_steps,1);
});
