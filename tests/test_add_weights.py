# TODO: update add weights tests
# import os


# def _test_add_weights(model, tmp_path, base_weights, added_weights, **kwargs):
#     from bioimageio.core.build_spec import add_weights

#     rdf = load_raw_resource_description(model)
#     assert base_weights in rdf.weights
#     assert added_weights in rdf.weights

#     weight_path = load_description(model).weights[added_weights].source
#     assert weight_path.exists()

#     drop_weights = set(rdf.weights.keys()) - {base_weights}
#     for drop in drop_weights:
#         rdf.weights.pop(drop)
#     assert tuple(rdf.weights.keys()) == (base_weights,)

#     in_path = tmp_path / "model1.zip"
#     export_resource_package(rdf, output_path=in_path)

#     out_path = tmp_path / "model2.zip"
#     add_weights(in_path, weight_path, weight_type=added_weights, output_path=out_path, **kwargs)

#     assert out_path.exists()
#     new_rdf = load_description(out_path)
#     assert set(new_rdf.weights.keys()) == {base_weights, added_weights}
#     for weight in new_rdf.weights.values():
#         assert weight.source.exists()

#     test_res = _test_model(out_path, added_weights)
#     failed = [s for s in test_res if s["status"] != "passed"]
#     assert not failed, failed
#     test_res = _test_model(out_path)
#     failed = [s for s in test_res if s["status"] != "passed"]
#     assert not failed, failed

#     # make sure the weights were cleaned from the cwd
#     assert not os.path.exists(os.path.split(weight_path)[1])


# def test_add_torchscript(unet2d_nuclei_broad_model, tmp_path):
#     _test_add_weights(unet2d_nuclei_broad_model, tmp_path, "pytorch_state_dict", "torchscript")


# def test_add_onnx(unet2d_nuclei_broad_model, tmp_path):
#     _test_add_weights(unet2d_nuclei_broad_model, tmp_path, "pytorch_state_dict", "onnx", opset_version=12)
