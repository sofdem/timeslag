# timeslag
Times model and Lagrangian relaxation

Solving the ETSAP-TIMES of the European power system through lagrangian relaxation, with subgradient or proximal bundle algorithms 
(based on Welington de Oliveira's code: https://sites.google.com/site/wdeolive/solvers), after dualizing the inter-region coupling constraints: 
the one-to-one interconnection constraints and the global CO2 limitation. 
A simple heuristic can be called at each iteration to derive a primal solution, by solving the primal LP model with some variables fixed according to the dual solution.

It requires Gurobi or Cplex for solving the linear programs. 
Problem instances: 2-region (France-Spain) or 29-regions on short- or long-time horizon.

Preliminary results published in Gildas Siggini's PhD thesis "Approche intégrée pour l'analyse prospective de la décarbonisation profonde du système électrique européen à l'horizon 2050 face à la variabilité climatique" (2021) supervised by Edi Assoumou and Sophie Demassey.
https://www.theses.fr/2021UPSLM010
