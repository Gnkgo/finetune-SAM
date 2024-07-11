import os
import yaml

requiremnts = open("requirements.txt", "w")
with open("environment.yml") as file_handle:
    environment_data = yaml.safe_load(file_handle)

for dependency in environment_data["dependencies"]:
    if isinstance(dependency, dict):
      for lib in dependency['pip']:
        print("CHECK")
        requiremnts.write(f"{lib}\n")