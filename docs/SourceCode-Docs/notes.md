The Code Components to inspect:
System, Environment (SolutionstepEnvironment,JointPRStepEnvironment),solver, logger, Counter, Controller, recorder 

The Flow :
Everything is within the sysstem instance (e.g. OnlineSystem)
system.run() method is called, inside which ready is called (for Training for learning solvers which is called pretrain in the code) then for number of experiments (each  experminet is a complete inference cycle) , in each one, the  env is reset, which returns an instance which is obs which is  a dict. then the solver solves the instance, returning a solution, the solution is given to the .step method of the env, returning next instance and a done flage  + info. if done is true, the experiment is done (we have reached to the  end of the VN stream)



TopologicalMetrics (TopologicalMetricCalculator)
AttributeBenchmarks (AttributeBenchmarkManager)
add_to_cache method 

ranking strategy (node ranking, link ranking )
node mapping (matching method, shortest_method, k_shortest)

solution.from_v_net




recorder.reset()
recorder.count_init_p_net_info(p_net)
recorder.add_record(record(it is a dict))
recorder.count(self.v_net, self.p_net, self.solution)



counter.count_solution(self.v_net, self.solution)

controller.deploy(self.v_net, self.p_net, self.solution)

self.controller.deploy(self.v_net, self.p_net, self.solution)
controller.release(self.v_net, self.p_net, solution)
total_p_resource_1_n = self.counter.calculate_sum_node_resource(self.p_net)
total_p_resource_1_e = self.counter.calculate_sum_link_resource(self.p_net)
total_p_resource_0_n = self.counter.calculate_sum_node_resource(self.p_net_backup)
total_p_resource_0_e = self.counter.calculate_sum_link_resource(self.p_net_backup)
total_p_resource_1 = self.counter.calculate_sum_network_resource(self.p_net)
total_p_resource_0 = self.counter.calculate_sum_network_resource(self.p_net_backup)








virne/network/
├── base_network.py                    → BaseNetwork(nx.Graph)
├── physical_network.py                → PhysicalNetwork(BaseNetwork)
├── virtual_network.py                 → VirtualNetwork(BaseNetwork)
├── virtual_network_request_simulator.py → VirtualNetworkRequestSimulator, VirtualNetworkEvent
├── dataset_generator.py               → Generator (orchestrates p_net + v_net dataset creation)
├── attribute/                         → BaseAttribute + Node/Link/Graph attribute subclasses
└── topology/                          → TopologyGenerator, TopologicalMetricCalculator





       Now I have to Recorder, Counter, Controller and then solver + (PRJOINTenv )








I havent specified yet the evaluation metrics: How R2C is calculated: I think:

for online VNE setting, the R2C is calculated after the whole simulation is completed, it says: takes those  VN requests have been embedded. they are embedded without an migration after imposed by node/link failures (called lucky VNs) or they are embedded with migration after(unlucky VNs). now take the revenues of each one. it is calculated as it was in virne. sum the revenues. call it R. now take the cost of each VN. for lukcy, it is the same as it is defined in virne. for unlucky, they same  +  migration cost + downtime cost + rerouting cost. sum the costst, call it C. the R2C is R/C.right? 

for cthe case we have no migration modeul, any node/link failure , since we cant migrate them, what kind of cost should be consider? (these two cases are supposeed to get compared in which the one with migration provides more R2C , acceptance ratio (the cost can be see this way that since those VNs once get embedded but didnt survive till the end of their liftime,  there should be no revenue considered for it but also a cost for it also should be consider since a period of time it consumed the physical network.... how to reformulate and articulate it?))




Where i need to refactor for migration module :

1- BaseSystem.from_config(config) - in  p_net, v_net_simulator = cls.load_dataset(logger, config). i need to load p_net_failure_simulator
