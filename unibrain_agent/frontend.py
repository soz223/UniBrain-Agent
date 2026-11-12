import json

from .paths import PROJECT


def document(payload=None, standalone=False):
    web = PROJECT / "web"
    template = (web / "dashboard.html").read_text(encoding="utf-8")
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    from .catalog import catalog
    from .roles import role_catalog, prompt_for
    from .data_contract import model_contract
    config = {"data_contract": model_contract(), "standalone": standalone, "catalog": catalog(), "roles": role_catalog(), "research_prompt": prompt_for("research")}
    parts = {
        "__STYLE__": "\n".join((web / name).read_text(encoding="utf-8") for name in ("dashboard.css", "assistant.css", "settings.css", "data.css", "cycle.css")),
        "__CONNECTION_SCRIPT__": (web / "connection.js").read_text(encoding="utf-8"),
        "__SCRIPT__": (web / "dashboard.js").read_text(encoding="utf-8"),
        "__EDITOR_SCRIPT__": (web / "editor.js").read_text(encoding="utf-8"),
        "__ASSISTANT_SCRIPT__": "\n".join((web / name).read_text(encoding="utf-8") for name in ("assistant.js", "settings.js", "data.js", "cycle.js")),
        "__ASSISTANT_PANELS__": "\n".join((web / name).read_text(encoding="utf-8") for name in ("assistant.html", "settings.html")),
        "__APP_CONFIG__": json.dumps(config, ensure_ascii=False).replace("<", "\\u003c"),
        "__REPORT_JSON__": encoded,
        "__DATA_PANEL__": (web / "data.html").read_text(encoding="utf-8"),
        "__CYCLE_PANEL__": (web / "cycle.html").read_text(encoding="utf-8"),
    }
    for key, value in parts.items():
        template = template.replace(key, value)
    for name in ("openai", "claude", "anthropic"):
        template = template.replace("__LOGO_" + name.upper() + "__", (web / "logos" / (name + ".svg")).read_text(encoding="utf-8"))
    return template
