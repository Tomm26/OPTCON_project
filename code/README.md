# Optimal Control of a Flexible Robotic Arm

The current folder contains all the code related to the development of optimal control for a Flexible Robotic Arm, created following the guidelines in the file [_Assignment OptimalControl FlexibleRoboticArm_][1]. For a more detailed explanation of the various methods contained in the files, refer to their content and the comments and descriptions included within them. Even more, for a better insight into the project, refer to the related Final Report.

### Main File Usage

The file _main.py_ is the most important file for running the code and observing its results.<br>
If you are running the project for the first time, the only thing you have to do is to run the _main.py_ file as is, and then observe the results!<br>

The **tasks list** at the beginning of the file can and must be used to select which tasks to execute and display in their results through appropriate plots.
The **lazyness list** at the beginning of the file can be used to indicate, for each executed task, whether it should attempt to retrieve the most recent save (if available) and load its data, or perform a new and clean execution of the code. Indeed, you should not rely on the lazyness mechanics if you have changed some project parameters (whether they are global parameters, contained in the file _parameters.py_, or local parameters, contained in the various tasks) with the aim of observing the new different results obtained from executing one or more set-to-active tasks.<br>
Afterwards, the code of the file _main.py_ proceeds to execute and manage the requested tasks and display their results, logging also information about the code execution as it evolves. Furthermore, each task, once its execution is completed, saves the obtained results to file.<br>

**An important first note**: be aware that some tasks reference to (and make use of) previous tasks. For example, both Task3 and Task4 retrieve the reference input-state trajectory from Task2 (which is executed in a lazy way if its lazyness is set to True).<br>
**A second important note**: to potentially modify the behavior of any one of the tasks, you have to refer to the variables contained in the respective Python code file (so task1.py for Task1, task2.py for Task2, and so on...). For example, in Task3 and Task4, to eventually add/remove any disturbances to the initial state or to activate/deactivate the generation of measurement noise along the whole trajectory, you have to modify the respective variables _xx0disturbanceLevels_ and _generateMeasureNoises_ within the files task3.py and task4.py respectively.<br>
**A third important note**: the execution of Task1 and Task2 may experience slowdowns due to the generation (for each iteration of the Newton Method) of Armijo plots that are good presentable to the eye (therefore populated with a sufficiently high number of points). To avoid this behavior, thereby achieving faster executions at the expense of generating simpler Armijo plots, the variable _generateNicePlots_ within the files _task1.py_ and _task2.py_ should be set to False. On the other hand, if you want nice Armijo plots to be generated, set that variable to True.

### Other files brief description

Here is a descriptive list of the other files that make up the project:
- **task1**: from a desired quasi-step state-input curve (that evolves from one equilibrium to another) to an optimal trajectory thanks to the regularized Newton's Like Method (in its closed-loop version)
- **task2**: from a desired smooth state-input curve (that evolves from one equilibrium to another) to an optimal trajectory thanks to the regularized Newton's Like Method (in its closed-loop version)
- **task3**: after linearizing the dynamics of the FRA around a given trajectory, exploiting the LQR algorithm to define the optimal feedback controller to track the said trajectory
- **task4**: after linearizing the dynamics of the FRA around a given trajectory, exploiting an MPC algorithm to track the said trajectory
- **task5**: producing an animation of the robot executing Task 3
- **solver**: solver for an Optimal Control Trajectory Generation Problem (and ALL the involved functions, as the N.M., or the Armijo StepSize selection rule, or the solver of an LQP, and so on)
- **dynamics**: this file contains the Flexible Robotic Arm discretized dynamics and some related methods, such as a method to evolve forward-in-time (in an open loop fashon) a dynamic system or also a method to compute a local linearization of a NL dynamic around a given trajectory
- **costs**: definition of the cost functions for a Trajectory Generation Optimal Control Problem
- **curves**: curves generator for states and inputs (in a fashion that a given series of points are properly connected)
- **equilibria**: Flexible Robotic Arm Equilibrium Points Searcher
- **regulators**: collection of the functions that implement the Regulators used in the Project (and correlated functions), alias LQR and MPC
- **miscellaneous**: this file contains some useful functions (and the classes TrjTrkOCPData and TrjTrkCntrlData, the first one used in Tasks 1 and 2 and the second one used in Task 3 and 4 to encapsulate generated data) that are used in the project
- **logger**: project logger, used in the whole project to print, alias log to the console, information about the execution
- **parameters**: this file contains all the parameters of the Flexible Robotic Arm (FRA) system, as well as other parameters used in the code
- **plots**: project plotter, used for all plots generation
- **animation**: project animator, to implement the Task5

[1]: https://github.com/VincenzoLomba/OptimalControl/blob/master/Data/Project/Assignment%20OptimalControl%20FlexibleRoboticArm.pdf
