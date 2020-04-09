# todo: 1) Refactoring names following PEP8
# todo: 2) Create processorRegistry format
# todo: 3) Create total pipeline
# todo: 4) Create first level models
# todo: 5) Create first level datasets
# todo: 6) Create first level train val test loops
# todo: 7) Create second level models
# todo: 8) Create second level datasets
# todo: 9) Create second level train val test loops
# todo: 10) Create tensorboard
# todo: 11) Create logging
# todo: 12) Create UNIT tests
# todo: 13) Create README
# todo: 14) Create docstrings

import importlib
import traceback
import time
import utils.mainUtils as mainUtils
import utils.misc

PROCESSOR_REGISTRY_CONFIG = "./processorRegistry.json"
CONSTANS = "./constants.json"


def run(processor_registry_config_path, constants_path):
    constants_config = utils.misc.read_json(constants_path)
    processor_registry_config = utils.misc.read_json(processor_registry_config_path)

    for processor in processor_registry_config:
        processor_data_types = mainUtils.get_processor_data(processor["input"]["data"], constants_config)
        processor_output_data_types = mainUtils.get_processor_data(processor["output"]["data"], constants_config)
        processor_input_config = mainUtils.get_processor_data(processor["input"]["config"], constants_config)

        is_output_existed = mainUtils.is_processor_output_created(processor_output_data_types)

        if processor["forceCreate"]=="False" and is_output_existed:
            print(f"INFO: For {processor['name']} output is existed")
            continue

        timestamp = time.time()
        processor_module = importlib.import_module(processor['module'])

        # run processor
        print(f"INFO: Running  {processor['name']}")
        try:
            processor_module.process(processor_data_types, processor_input_config, processor_output_data_types)
        except Exception as e:
            print(f"ERROR: Exception during processor {processor['name']} execution: ", e)
            traceback.print_exc()
            print(f"INFO: {processor['name']} failure time = {str(time.time() - timestamp)}")
            break

        print(f"INFO: Processor \"{processor['name']}\" execution time = {str(time.time() - timestamp)}")

    print("Done!")


if __name__ == "__main__":
    run(PROCESSOR_REGISTRY_CONFIG, CONSTANS)

