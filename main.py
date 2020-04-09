import importlib
import traceback
import time
import utils.mainUtils as mainUtils
import utils.misc

PROCESSOR_REGISTRY_CONFIG = "./processorRegistry.json"
CONSTANS = "./constants.json"


def run(processor_registry_config_path, constants_path):
    """
    Main manager of processors. Run all processors from processorRegistry.json consistently

    Args:
        processor_registry_config_path: path processorRegistry.json
        constants_path: path constants.json

    """

    # read configs
    constants_config = utils.misc.read_json(constants_path)
    processor_registry_config = utils.misc.read_json(processor_registry_config_path)

    # main processors loop
    for processor in processor_registry_config:

        # get processors data
        processor_data_types = mainUtils.get_processor_data(processor["input"]["data"], constants_config)
        processor_output_data_types = mainUtils.get_processor_data(processor["output"]["data"], constants_config)
        processor_input_config = mainUtils.get_processor_data(processor["input"]["config"], constants_config)

        is_output_existed = mainUtils.is_processor_output_created(processor_output_data_types)

        # check on force run processor even data is existed. If data is existed then also skip processor
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

