import random
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import pandas as pd
import warnings

class Person:
    
    def __init__(self, Weight: float = 0.7, debug_id: int = 0, agentSIR: bool = False, SirTime: int = 0) -> None:
        self.person_memory = []
        self.debugId = debug_id
        self.Weight = Weight # Memory weight
        self.ImInfected = False # Rappresente an answer to the qestion: Am I infected?
        self.levelContagious = 0 # rappresent the contagious threshold
        self.timeContagious = 0 # rappresent how much time the agent is contagious
        self.ContagiousWillStopAt = 0 # rappresent the remaning time for how much this agent remains contagious
        self.infectionStartingWeek = 0 # Rappresent the week when the infection has started
        self.SIRWillStopAt = -1
        if SirTime == 0:
            self.considerSirTime = False
            self.SirTime = 0
        else:
            self.considerSirTime = True
            self.SirTime = SirTime
        self.agentSIR = agentSIR
        self.SIR_infectionsCounter = 0

    
    def personStrategyOutput(self) -> None: # This function calculates the latest strategy for the agents every week
        s_output = random.random()
        self.person_memory.append(s_output)
    
    def memoryMean(self) -> float: # This function returns the actual strategy composed by the last strategy and the ol strategies of the past weeks. This is done through a weighed average.
        if len(self.person_memory) > 1:
            last_s = self.person_memory[len(self.person_memory) - 1]
            r_Weight = (1 - self.Weight)
            sum_n = 0 
            sum_d = 0
            for i in range(0, len(self.person_memory) - 1):
                x = self.person_memory[i]
                sum_n += (x * r_Weight)
                sum_d += r_Weight
            sum_n += (last_s * self.Weight)
            sum_d += self.Weight
            return sum_n / sum_d
        
        else:
            return self.person_memory[0]
    
    def personCurrentStrategy(self) -> float: # This function is a wrapper for personStrategyOutput and memoryMean
        self.personStrategyOutput()
        return self.memoryMean()
    
    def updateLastStrategy(self, real_n) -> None: # In case an agent goes to the bar, this function updates the strategy of the agent
        self.person_memory.pop(len(self.person_memory) - 1)
        self.person_memory.append(real_n)
    
    def initiateContagius(self, contagiousTime: int, infectionStartingWeek: int) -> bool: # This function is needed to initiate the contagious for the agent (returns a boolean that indicates if the contagious was executed)
        infectAgentDecisionSIR = False
        if self.agentSIR:
            if self.considerSirTime:
                if infectionStartingWeek >= self.SIRWillStopAt:
                    infectAgentDecisionSIR = True
                    if self.debugId == 1: # Used to view info of certain agents
                        print('Sir stopping at week: %d' % (self.SIRWillStopA + 1))
            else:
                if self.SIR_infectionsCounter == 0:
                    infectAgentDecisionSIR = True
        else:
            infectAgentDecisionSIR = True

        if infectionStartingWeek == -1:
            infectAgentDecisionSIR = True
                
        if infectAgentDecisionSIR:
            self.levelContagious = 1  # This rapresent the initial contagious level
            self.timeContagious = contagiousTime # This rapresent for how much time the agent will remain infected
            if self.debugId == 1: # Used to view info of certain agents
                pass
            self.ContagiousWillStopAt = self.timeContagious + infectionStartingWeek # This rapresent the remaining time for the agent to remain infected, the first day this is equal to self.timeContagious
            self.ImInfected = True # This is a boolean that indicates if the agent is infected

            self.infectionStartingWeek = infectionStartingWeek # This indicate the number of the week in which the agent is infected
            
            self.SIR_infectionsCounter += 1 # Counts how many time the agent gets infected

            if self.agentSIR:
                if self.considerSirTime:
                    self.SIRWillStopAt = self.SirTime + self.ContagiousWillStopAt + 1
            return True
        else:
            return False


    def getContagiousLevel(self, current_week: int = -1): # This functions calculates the level of contagious for the agents every week
        if current_week == -1:
            raise Exception('Contagious level error')
            
        if self.ContagiousWillStopAt - current_week >= 0:
            if self.debugId == 1: # Used to view info of certain agents
                print(self.levelContagious, 'Infected: ' + str(self.ImInfected), self.ContagiousWillStopAt, current_week)
            x, x_max = 0.1, 1
            x_ts = [x]
            t_max = self.timeContagious
            c, d = 0.5, 0.4

            t = (current_week - self.infectionStartingWeek)

            x = x + c * (1 - x / x_max) 
            x = x * ( 1 - t / t_max )
            
            self.levelContagious = x

            return x
            
        else:
            if self.ImInfected:
                self.ImInfected = False
                self.levelContagious = 0
                self.ContagiousWillStopAt = 0
                self.infectionStartingWeek = 0
                self.SIRWillStopAt = self.SirTime + current_week
                if self.debugId == 1: # Used to view info of certain agents
                    print('Infected: ' + str(self.ImInfected))
            return 0

    def getIfInfected(self): # This function returns a boolean that indicates if the agent is infected
        return self.ImInfected
    

class ElFarolBar:
    def __init__(self, seed, num_agents, num_contagious_agents, contagiousness, capacity, threshold, contagious_threshold, contagious_duration, people_memory_weight: float = 0.7, contagious_thresholdNotPresent: float = 1, Use_SIR: bool = False, SIR_AgentsRecoveryTime: int = 0, debugCSV: bool = False): 
        self.seed = seed # This is a random seed used to generate random numbers, to guarantee replicability
        self.num_agents = num_agents
        self.num_contagious_agents = num_contagious_agents # Identifies the number of starting contagious agents
        self.agents = [] # This arra contains all the agents needed fot the simulation
        self.contagiousness = contagiousness
        self.capacity = capacity # This integer represent the maximum capacity of the bar (is userfull if respect_the_max: bool = True)
        self.threshold = threshold # This threshold is used to determine if an agent will go to the bar or not depending on his strategy
        self.contagious_threshold = contagious_threshold # Another person gets contagious if his contagious level is greather than contagious_threshold
        self.contagious_duration = contagious_duration # An agent is contagious for contagious_duration weeks
        self.attendance_history = [] # This array is composed from a series of integers rapresenting the number of people in the bar
        self.contagious_history = [] # This array is composed from a series of integers rapresenting the number of contagious people
        self.present_contagious_history = [] # This array is composed from a series of integers rapresenting the number of contagious people in the bar
        self.people_memory_weight = people_memory_weight # This is a parameter that belongs to the agents
        self.contagious_thresholdNotPresent = contagious_thresholdNotPresent # This is a parameter that belongs to the agents
        self.debugBool = False # This boolean is used to enable console debug (enable at line 137-138)
        self.debugCSV = debugCSV
        self.debugCSVFolderName = ""
        self.debugCSVPresentName = ""
        self.debugCSVAgentStratLevelName = ""
        self.debugCSVPath = ""

        # Making CSV paths
        if self.debugCSV: # This will generate the paths where to save csv file for export
            CSVSaveN = len(os.listdir('OutputCSV/'))
            self.debugCSVFolderName = 'ExportCSV_' + str(CSVSaveN)
            self.debugCSVPath = 'OutputCSV/' + self.debugCSVFolderName
            os.mkdir(self.debugCSVPath)
            self.debugCSVPresentName = self.debugCSVPath + '/ExportCSV_' + str(CSVSaveN) + '_PresentBool.csv'
            self.debugCSVAgentStratLevelName = self.debugCSVPath + '/ExportCSV_' + str(CSVSaveN) + '_AgentStrat.csv'
            self.debugCSVAgentCLevelName = self.debugCSVPath + '/ExportCSV_' + str(CSVSaveN) + '_AgentCLevel.csv'
            self.debugXLSXExportFileName = self.debugCSVPath + '/ExportXLSX_' + str(CSVSaveN) + '.xlsx'

        self.exportArr_PresentBool = []
        self.exportArr_AgentStrat = []
        self.exportArr_AgentCLevel = []


        # Building agents objects
        for i in range(self.num_agents): # This generates the agents needed for the simulation
            if i == 0:
                person = Person(self.people_memory_weight, agentSIR=Use_SIR, SirTime=SIR_AgentsRecoveryTime, debug_id=0) # <- Set to 1 to enable debug on one agent
                self.debugBool = False # <- Set to True to enable debug
            else:
                person = Person(Weight=self.people_memory_weight, agentSIR=Use_SIR, SirTime=SIR_AgentsRecoveryTime)
            if (i + 1) <= num_contagious_agents: 
                person.initiateContagius(self.contagious_duration, infectionStartingWeek=-1)
            self.agents.append(person)
            


    def simulate(self, num_weeks: int, respect_the_max: bool = False):
    
        random.seed(self.seed) # This is used to set the seed of the random generator

        for week in range(num_weeks): # This cycle performs the simulation for all the weeks
            attendance = 0 # This integer rapresent the n of the agents going to the bar
            present_agents = [] # This is the array containing al the agents that will be in the bar each week
            infected_attendance = 0 # This integer rapresent the n of the agents which are infected each week

            # The following dictionaries are used to export data into CSV
            self.exportDict_PresentBool = {}
            self.exportDict_AgentStrat = {}
            self.exportDict_AgentCLevel = {}

            # Calculating how many agents are going to the bar
            for i, agent in enumerate(self.agents):

                a_strat = agent.personCurrentStrategy() # This float rapresent the strategy of agent each week
                
                c_level = 0 # This float rapresent the contagious level of agent each week
                if agent.getIfInfected(): # If agent is infected
                    c_level = agent.getContagiousLevel(current_week=week) # Calculate contagious level
                    self.exportDict_AgentCLevel [i + 1] = c_level 
                else:
                    self.exportDict_AgentCLevel [i + 1] = 0
                    
                if agent.debugId == 1: print('Agent current strategy: %.3f' % a_strat)

                if a_strat < self.threshold and c_level <= self.contagious_thresholdNotPresent: # If the agent strategy for this week and contagious level is below the not present threshold, he will be in the bare.
                    attendance += 1
                    present_agents.append(agent)
                    self.exportDict_PresentBool [i + 1] = 1
                else:
                    self.exportDict_PresentBool [i + 1] = 0

                self.exportDict_AgentStrat [i + 1] = a_strat
                
            # If respect the max i True -> maximum n of agents is <= self.capacity
            if respect_the_max:
                if attendance >= self.capacity:
                    attendance = self.capacity

            # Updating strategy of the present agents
            if respect_the_max:
                present_agents_strategy = attendance / self.capacity
            else:
                present_agents_strategy = attendance / self.num_agents
            for present_agent in present_agents:
                present_agent.updateLastStrategy(present_agents_strategy)


            # Let the infection start 
            n_new_infected = 0 # Integer rapresenting the n of agents that will be infected in the current week
            contagious_level_sum = 0 # Float rapresenting the sum of contagious level for all the infected agents present in the bar
            n_infected_agents = 0 # Integer rapresenting the n of infected agents in the bar
            for present_agent in present_agents: # This will calculate contagious_level_sum
                contagious_level_sum += present_agent.levelContagious

            for present_agent in present_agents: # This will calculate n_infected_agents
                if present_agent.getIfInfected():
                    n_infected_agents += 1
                
            n_susceptible_agents = len(present_agents) - n_infected_agents # This is the n of agents that could be infected this week
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    n_new_infected = int(self.contagiousness * contagious_level_sum * n_susceptible_agents / (n_susceptible_agents + n_infected_agents)) # This is the n of agents that will be infected this week
                    # print(n_new_infected, int(contagious_level_sum * n_susceptible_agents / (n_susceptible_agents + n_infected_agents)))
            except:
                # print('Error week %d, divison0 %.2f' % (week, (n_susceptible_agents + n_infected_agents)))
                # print(f'Error - contagiousness: {self.contagiousness}, n_susceptible_agents: {n_susceptible_agents}, n_infected_agents: {n_infected_agents}, contagious_level_sum: {contagious_level_sum}')
                n_new_infected = 0

            totInfectedWeekByAgent = 0 # This is a counter for people infected this week by each agent
            for present_infectious_agent in present_agents: # For each infectous agents between the present agents
                totInfectedWeekByAgent = n_new_infected
                if present_infectious_agent.levelContagious >= self.contagious_threshold and present_infectious_agent.levelContagious <= self.contagious_thresholdNotPresent: # If the agent can infect other agents
                    for present_agent in present_agents: # For each present agent
                        if present_agent.getIfInfected() == False: # If not infected
                            if totInfectedWeekByAgent > 0: # InitiateContagious for n_new_infected agents
                                contagious_execution = present_agent.initiateContagius(self.contagious_duration, week) # InitiateContagious of present agent
                                if contagious_execution:
                                    totInfectedWeekByAgent -= 1 # The counter for the people that could be infected by de agent, decrease by one

            n_infected_agents = 0                
            for present_agent in present_agents: # Updating n_infected_agents
                if present_agent.getIfInfected():
                    n_infected_agents += 1
            self.present_contagious_history.append(n_infected_agents)
            
                        
            # Calculating n of infected agents
            for agent in self.agents: 
                if agent.getIfInfected():
                    infected_attendance += 1
                    if agent.debugId == 1:
                        if agent.infectionStartingWeek == week:
                            print(str(agent.infectionStartingWeek) + '<- Debug Agent infected. ', str(week) + '<- Current week')
                        else:
                            print(str(week - agent.infectionStartingWeek) + '<- Weeks being infected for Agent 1')
                        
            
            if self.debugBool:
                print('\n\nWeek: %d' % (week + 1))
            
            self.attendance_history.append(attendance)
            self.contagious_history.append(infected_attendance)

            self.exportArr_PresentBool.append(self.exportDict_PresentBool)
            self.exportArr_AgentStrat.append(self.exportDict_AgentStrat)
            self.exportArr_AgentCLevel.append(self.exportDict_AgentCLevel)

            if infected_attendance == 0 and week >= 3: break

        #Writing CSV export data
        if self.debugCSV:

            #Present bool csv
            with open(self.debugCSVPresentName, mode='w') as csvfile:
                fieldnames = self.exportArr_PresentBool[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.exportArr_PresentBool)

            #Agent strat csv
            with open(self.debugCSVAgentStratLevelName, mode='w') as csvfile:
                fieldnames = self.exportArr_AgentStrat[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.exportArr_AgentStrat)

            #Agent contagious level csv
            with open(self.debugCSVAgentCLevelName, mode='w') as csvfile:
                fieldnames = self.exportArr_AgentCLevel[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.exportArr_AgentCLevel)
                
            #Combining CSV files
            try:
                exportFilesToMerge = os.listdir(self.debugCSVPath)
                with pd.ExcelWriter(self.debugXLSXExportFileName) as writer:
                    for ExportFile in exportFilesToMerge:
                        exportFileToMerge = self.debugCSVPath + '/' + ExportFile
                        df = pd.read_csv(exportFileToMerge)
                        sheetName = os.path.splitext(ExportFile)[0].split('_')
                        df.to_excel(writer,sheet_name= sheetName[len(sheetName) - 1])

            except:
                print('No module openpyxl installed.\nThis is necessary in order to merge CSV export file into one xlsx file.\nIn order to intall openpyxl run: "pip install openpyxl"')

        return [week, self.attendance_history, self.contagious_history, self.present_contagious_history]
        
    
    def chart(self, max_line: bool = True, threshold_line: bool = True, cont_threshold_line: bool = True, contNotPres_threshold_line: bool = True):
        arr_x = []
        for i in range(1, len(self.attendance_history) + 1):
            arr_x.append(i)
        plt.plot(arr_x, self.attendance_history, label="People in the bar every week")
        plt.plot(arr_x, self.contagious_history, label="Contagious people (%d weeks)" % self.contagious_duration)
        plt.plot(arr_x, self.present_contagious_history, label="Contagious people level in bar")
        if max_line:
            line_max = plt.Line2D((0, arr_x[len(arr_x) - 1]), (self.capacity, self.capacity), color="red", label=f"Maximum capacity: {self.capacity}")
            plt.gca().add_line(line_max)
        if threshold_line:
            line_threshold = plt.Line2D((0, arr_x[len(arr_x) - 1]), (self.threshold * self.num_agents, self.threshold * self.num_agents), color="green", label=f"Agents treshold: {self.threshold * self.num_agents}")
            plt.gca().add_line(line_threshold)
        if cont_threshold_line:
            line_threshold = plt.Line2D((0, arr_x[len(arr_x) - 1]), (self.contagious_threshold * self.num_agents, self.contagious_threshold * self.num_agents), color="blue", label=f"Contagious treshold: {self.contagious_threshold * self.num_agents}")
            plt.gca().add_line(line_threshold)
        if contNotPres_threshold_line:
            line_threshold = plt.Line2D((0, arr_x[len(arr_x) - 1]), (self.contagious_thresholdNotPresent * self.num_agents, self.contagious_thresholdNotPresent * self.num_agents), color="lightblue", label=f"Contagious treshold (not present): {self.contagious_thresholdNotPresent * self.num_agents}")
            plt.gca().add_line(line_threshold)
        plt.legend(loc="upper left", bbox_to_anchor=(-0.15,1.25), ncol=2)
        plt.tight_layout()
        plt.show()

    def chartSave(self, max_line: bool = True, threshold_line: bool = False, cont_threshold_line: bool = False, contNotPres_threshold_line: bool = False, experiment : str = ""):
        
        fig, ax = plt.subplots(figsize=(10,6))
        
        arr_x = []
        for i in range(1, len(self.attendance_history) + 1): arr_x.append(i)
        if max_line:
            line_max = plt.Line2D((0, arr_x[len(arr_x) - 1]), (self.capacity, self.capacity), color="black", linestyle='--', label="Maximum capacity")
            plt.gca().add_line(line_max)    
        ax.plot(arr_x, self.attendance_history, label="Attendance", c= 'navy')
        ax.plot(arr_x, self.contagious_history, label="Infected", c= 'salmon')
        ax.plot(arr_x, self.present_contagious_history, label="Infected attendance", c= 'salmon', linestyle='--', alpha = 0.4)
        if threshold_line:
            line_threshold = plt.Line2D((0, arr_x[len(arr_x) - 1]), (self.threshold * self.num_agents, self.threshold * self.num_agents), color="green", label="Agents treshold")
            plt.gca().add_line(line_threshold)
        if cont_threshold_line:
            line_threshold = plt.Line2D((0, arr_x[len(arr_x) - 1]), (self.contagious_threshold * self.num_agents, self.contagious_threshold * self.num_agents), color="blue", label="Contagious threshold")
            plt.gca().add_line(line_threshold)
        if contNotPres_threshold_line: 
            line_threshold = plt.Line2D((0, arr_x[len(arr_x) - 1]), (self.contagious_thresholdNotPresent * self.num_agents, self.contagious_thresholdNotPresent * self.num_agents), color="lightblue", label="Contagious treshold (not present)")
            plt.gca().add_line(line_threshold)
        ax.legend(loc="upper left", bbox_to_anchor=(-0.01,1.15), ncol=2)
        file_path = os.path.abspath(__file__)
        folder_path = os.path.dirname(file_path)
        baseDir = "/OutputImg/"
        fig.savefig(folder_path + baseDir + "Export_" + experiment + ".png")
        #plt.savefig(folder_path + baseDir + "Export_" + experiment + ".png", bbox_inches="tight")
        plt.close(fig)



    def resultTest(self):
        arr_x = []
        for i in range(1, len(self.attendance_history) + 1): arr_x.append(i)

        plt.plot(arr_x, self.attendance_history, label="People in the bar every week")
        plt.plot(arr_x, self.contagious_history,label="Contagious people (%d weeks)" % self.contagious_duration)
        plt.plot(arr_x, self.present_contagious_history,label="Contagious people level in bar")
        plt.legend()
        plt.show()