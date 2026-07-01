from spebt_agent.assets.registry import list_model_assets, list_tool_environments


def test_tool_env_names_are_scoped():
    envs = list_tool_environments()
    assert envs
    assert all(env.env_name.startswith("spebt_") for env in envs)
    assert {env.module for env in envs} >= {"brightness", "stability", "submission"}


def test_model_assets_have_sizes_and_targets():
    assets = list_model_assets()
    assert assets
    assert all(asset.size_bytes > 0 for asset in assets)
    assert all(asset.target for asset in assets)
