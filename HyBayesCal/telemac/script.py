from HyBayesCal.telemac.control_telemac import TelemacModel
import sys
class TelemacSimulations:
    def single_run_simulation(self):
        """
        Executes a single Telemac simulation based on command-line arguments.
        :return: None
        """
        if len(sys.argv) != 9:
            print(len(sys.argv))
            print("Incorrect number of command-line arguments passed to the script!!")
            sys.exit(1)
        self.i = sys.argv[1]  # 2
        self.case_file = str(sys.argv[2])  # 't2d-donau-1.cas'
        self.result_filename_path = str(sys.argv[3]) # '/home/amintvm/modeling/hybayescalpycourse/examples/donau/r2d-donau-1.slf'
        self.results_filename_base = str(sys.argv[4]) # 'r2d-donau'
        self.tm_model_dir = str(sys.argv[5]) #'/home/amintvm/modeling/hybayescalpycourse/examples/donau/'
        self.tm_xd = str(sys.argv[6]) #'Telemac2d'
        self.N_CPUS = int(sys.argv[7]) # 1
        print(str(sys.argv[8]))
        self.CALIB_TARGETS = ','.join(part.replace('U', ' U').replace('V', ' V').replace('DEPTH', ' DEPTH').upper() for part in str(sys.argv[8]).split(".")) #['VELOCITY' 'DEPTH']
        self.CALIB_TARGETS = [value.strip() for value in self.CALIB_TARGETS.split(",")]
        print(type(self.CALIB_TARGETS))
        print(self.CALIB_TARGETS)
        tm_model = TelemacModel(
            model_dir=self.tm_model_dir,
            control_file=self.case_file,
            tm_xd=self.tm_xd,
            n_processors=self.N_CPUS,
            )
        tm_model.run_simulation()
        for calib_target in self.CALIB_TARGETS:
            tm_model.get_variable_value(slf_file_name=self.result_filename_path,
             calibration_par=calib_target, specific_nodes=None,
             save_name=self.tm_model_dir + f"/auto-saved-results/"
                                           f"{self.results_filename_base}-{self.i}_{calib_target}.txt"
             )
if __name__ == "__main__":
    simulation = TelemacSimulations()
    simulation.single_run_simulation()