
def get_ntok_list_per_instance(model):
    # get ntok list for each instance:
    output_stats = getattr(model.get_vision_tower().vision_tower, "output_stats", None)
    if output_stats is None or output_stats == {}:
        return None
    ntok_instances = []
    for i in range(len(output_stats["block_0_ntoks"])):
        ntok_instance = {k: int(v[i]) for k, v in output_stats.items()}
        ntok_instances.append(ntok_instance)
    return ntok_instances
