# import sys
# import subprocess

# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
# 'ortools'])
import ortools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time


def create_data_model(distance_matrix, demand, demandWC, num_bus, capacities, capacitiesWC):
    """Stores the data for the problem."""
    data = {}
    # dis_matrix = distance_matrix
    # dis_matrix = dis_matrix.round(0)
    # dist_lst = dis_matrix.values.tolist()
    # data['distance_matrix'] = dist_lst
    data['distance_matrix'] = distance_matrix#comment out!
    dist_lst=distance_matrix#comment out!
    data['demands'] = demand
    data['demandsWC'] = demandWC
    data['vehicle_capacities'] = capacities
    data['vehicle_capacities_WC'] = capacitiesWC
    data['num_vehicles'] = num_bus
    data['starts'] = [0] * num_bus
    data['ends'] = [len(dist_lst) - 1] * num_bus
    return data


def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []

    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes


def print_solution(data, manager, routing, solution, penalty):
    """Prints solution on console."""
    total_cost = 0
    total_load = 0
    num_vehicle_used = 0
    route_len_list = []
    load_list = []
    WC_load_list = []

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        route_load_wc = 0
        route_load_total = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            route_load_wc += data['demandsWC'][node_index]
            route_load_total += data['demands'][node_index] + data['demandsWC'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load_total)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load_total)
        plan_output += 'Time of the route: {}min\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load_total)
        #print(plan_output)

        if route_distance > 0:
            route_len_list.append(route_distance - penalty)
        else:
            route_len_list.append(route_distance)
        load_list.append(route_load)
        WC_load_list.append(route_load_wc)
        total_cost += route_distance
        total_load += route_load + route_load_wc
        if route_distance > 0:
            num_vehicle_used += 1
    #print('Total cost of all routes: {}min'.format(total_cost))
    #print('Total time of all routes: {}min'.format(total_cost - num_vehicle_used * penalty))
    #print('length of each route:', route_len_list)
    #print('Total total-load of all routes: {}'.format(total_load))
    #print('non-WC load for each route:', load_list)
    #print('WC load for each route:', WC_load_list)
    #print('Number of vehicles used : {}'.format(num_vehicle_used))

    return total_cost - num_vehicle_used * penalty, num_vehicle_used, route_len_list, load_list, WC_load_list


def main(dist_matrix, demand, demandWC, num_bus, capacities, capacitiesWC, penalty, ortime):
    """Solve the CVRP problem."""
    # Instantiate the data problem.

    data = create_data_model(dist_matrix, demand, demandWC, num_bus, capacities, capacitiesWC)
    penalties = [99999999]*len(dist_matrix) #penalty for not visiting a node
    #print('---Instantiated the data problem---')
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['starts'],
                                           data['ends'])
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)
    # add a fixed cost for each vehicle
    routing.SetFixedCostOfAllVehicles(penalty)
    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        50,  # vehicle maximum travel time 30mins
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    # distance_dimension.SetGlobalSpanCostCoefficient(100)
    distance_dimension.SetGlobalSpanCostCoefficient(0)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
    # Allow to drop nodes.
    #for node in range( len(penalties)):
        #routing.AddDisjunction([manager.NodeToIndex(node)], penalties[node])

    # Add WC Capacity constraint.
    def demandWC_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to WC demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demandsWC'][from_node]

    demandWC_callback_index = routing.RegisterUnaryTransitCallback(
        demandWC_callback)
    routing.AddDimensionWithVehicleCapacity(
        demandWC_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities_WC'],  # vehicle maximum WC capacities
        True,  # start cumul to zero
        'CapacityWC')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION

    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = ortime
    # search_parameters.log_search = True
    #print('Solving the problem.')
    # import time
    #start = time.process_time()

    solution = routing.SolveWithParameters(search_parameters)
    #print('solution',solution)
    #elapse = time.process_time() - start

    #print('Solved the problem.')
    #print('runtime: {}s'.format(elapse))
    # print(solution)
    # Print solution on console.
    print(solution)
    if solution:
        (total_distance, num_vehicle_used, route_len_list, load_list, WC_load_list) = print_solution(data, manager,
                                                                                                     routing, solution,
                                                                                                     penalty)
        routes = get_routes(solution, routing, manager)
        '''# Display dropped nodes.
        dropped_nodes = 'Dropped nodes:'
        for node in range(routing.Size()):
            if routing.IsStart(node) or routing.IsEnd(node):
                continue
            if solution.Value(routing.NextVar(node)) == node:
                dropped_nodes += ' {}'.format(manager.IndexToNode(node))
        print(dropped_nodes)'''
    else:
        print('----no solution found----')
        routes = [[0] * 2]
        num_vehicle_used = 0
        total_distance = 0
        route_len_list = [0 * 2]
        load_list = [0 * 2]
        WC_load_list = [0 * 2]
    #print('runtime', elapse)
    # print('Total distance of all routes: {}min'.format(total_distance-num_vehicle_used*penalty))
    return routes, total_distance, num_vehicle_used, route_len_list, load_list, WC_load_list


#testing:
'''time_matrix= [
        [
            0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354,
            468, 776, 662
        ],
        [
            548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674,
            1016, 868, 1210
        ],
        [
            776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164,
            1130, 788, 1552, 754
        ],
        [
            696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822,
            1164, 560, 1358
        ],
        [
            582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708,
            1050, 674, 1244
        ],
        [
            274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628,
            514, 1050, 708
        ],
        [
            502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856,
            514, 1278, 480
        ],
        [
            194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320,
            662, 742, 856
        ],
        [
            308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662,
            320, 1084, 514
        ],
        [
            194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388,
            274, 810, 468
        ],
        [
            536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764,
            730, 388, 1152, 354
        ],
        [
            502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114,
            308, 650, 274, 844
        ],
        [
            388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194,
            536, 388, 730
        ],
        [
            354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0,
            342, 422, 536
        ],
        [
            468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536,
            342, 0, 764, 194
        ],
        [
            776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274,
            388, 422, 764, 0, 798
        ],
        [
            662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730,
            536, 194, 798, 0
        ],
    ]
main(time_matrix,[0]*17,[0]*17,500,[50]*500,[0]*500,penalty=1000,ortime=1)'''
