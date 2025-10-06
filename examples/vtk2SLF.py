import sys,os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, 'src')
hydroBayesCal_path = os.path.join(src_path, 'hydroBayesCal')
sys.path.insert(0, base_dir)
sys.path.insert(0, src_path)
sys.path.insert(0, hydroBayesCal_path)

# Add the paths to sys.path
sys.path.insert(0, src_path)  # Prepend to prioritize over other paths
sys.path.insert(0, hydroBayesCal_path)

from src.hydroBayesCal.function_pool import vtk_to_2dm,twodm2SLF

input_vtk_file_path = os.path.join(base_dir,'examples/munichFishpass/test-structured.vtk')
output_2dm_path = os.path.join(base_dir,'examples/munichFishpass/test.2dm')
output_adcirc_path = os.path.join(base_dir,'examples/munichFishpass/test.grd')
output_SLF_path = os.path.join(base_dir,'examples/munichFishpass/test.slf')
vtk_to_2dm(input_vtk_file_path, output_2dm_path)
twodm2SLF(output_2dm_path,output_adcirc_path,output_SLF_path)
