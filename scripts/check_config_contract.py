

from __future__ import annotations

import ast
import configparser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST: dict[str, str] = {
    "model_config.frequency_penalty": "optional sampling penalty, documented as a comment in the template",
    "model_config.presence_penalty": "optional sampling penalty, documented as a comment in the template",
    "model_config.freq_penalty": "legacy alias the loader still accepts for frequency_penalty",
    "model_config.pres_penalty": "legacy alias the loader still accepts for presence_penalty",
    "ai_providers.blockrun_base_url": "BlockRun is reachable only inside the all-chain; the template keeps the pair commented out",
    "ai_providers.blockrun_model": "BlockRun is reachable only inside the all-chain; the template keeps the pair commented out",
    "executor_api.intent_journal_path": "optional override; the default is derived from [directories] data_dir",
}


class ConfigContractChecker:
    """Compare the keys the loader reads with the ones the template, the ini and the docs carry."""

    def __init__(self, root: Path = PROJECT_ROOT) -> None:
        self.root = root
        self.src_dir = root / "src"
        self.loader_path = root / "src" / "config" / "loader.py"
        self.example_path = root / "config" / "config.ini.example"
        self.active_path = root / "config" / "config.ini"
        self.docs_path = root / "docs" / "CONFIGURATION.md"
        self.valid_providers = {"local", "googleai", "openrouter", "deepseek", "all"}

    def _read_ini(self, path: Path) -> dict[str, set[str]]:
        parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
        parser.optionxform = lambda option: option
        parser.read(path, encoding="utf-8")
        return {
            section: {
                key if section == "model_context_windows" else key.lower()
                for key in parser[section]
            }
            for section in parser.sections()
        }

    def _loader_reads(self) -> list[tuple[str, str, str]]:
        reads: list[tuple[str, str, str]] = []
        for path in sorted(self.src_dir.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            relative = path.relative_to(self.root).as_posix()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Attribute) or node.func.attr != "get_config":
                    continue
                arguments = node.args
                if len(arguments) < 2:
                    continue
                section, key = arguments[0], arguments[1]
                if not isinstance(section, ast.Constant) or not isinstance(key, ast.Constant):
                    continue
                if not isinstance(section.value, str) or not isinstance(key.value, str):
                    continue
                reads.append((section.value, key.value.lower(), f"{relative}:{node.lineno}"))
        return reads

    def _required_keys(self) -> dict[str, tuple[str, ...]]:
        tree = ast.parse(self.loader_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "REQUIRED_CONFIG_KEYS" not in target_names or not isinstance(node.value, ast.Dict):
                continue
            required: dict[str, tuple[str, ...]] = {}
            for section, keys in zip(node.value.keys, node.value.values, strict=True):
                if not isinstance(section, ast.Constant) or not isinstance(keys, ast.Tuple):
                    continue
                required[str(section.value)] = tuple(
                    str(element.value) for element in keys.elts if isinstance(element, ast.Constant)
                )
            return required
        return {}

    def _missing_from_example(self, example: dict[str, set[str]]) -> list[str]:
        violations: list[str] = []
        for section, key, where in sorted(set(self._loader_reads())):
            if section not in example:
                violations.append(f"{where}: section [{section}] is missing from config.ini.example")
            elif key not in example[section] and f"{section}.{key}" not in ALLOWLIST:
                violations.append(
                    f"{where}: [{section}] {key} is read by the loader but missing from config.ini.example"
                )
        return violations

    def _unknown_in_active_config(self, example: dict[str, set[str]]) -> list[str]:
        if not self.active_path.exists():
            return []
        violations: list[str] = []
        active = self._read_ini(self.active_path)
        for section, keys in sorted(active.items()):
            for key in sorted(keys):
                if section not in example:
                    violations.append(
                        f"config/config.ini: section [{section}] does not exist in config.ini.example"
                    )
                elif key not in example[section] and f"{section}.{key}" not in ALLOWLIST:
                    violations.append(
                        f"config/config.ini: [{section}] {key} is not in config.ini.example; "
                        "a typo here is silently ignored by the loader"
                    )
        return violations

    def _required_in_example(self, example: dict[str, set[str]]) -> list[str]:
        violations: list[str] = []
        for section, keys in sorted(self._required_keys().items()):
            for key in keys:
                if section not in example or key not in example[section]:
                    violations.append(
                        f"REQUIRED_CONFIG_KEYS: [{section}] {key} is missing from config.ini.example"
                    )
        return violations

    def _documented_keys(self, example: dict[str, set[str]]) -> list[str]:
        if not self.docs_path.exists():
            return []
        known = {key for keys in example.values() for key in keys}
        violations: list[str] = []
        in_core_table = False
        for line in self.docs_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                in_core_table = stripped == "## Core settings"
                continue
            if not in_core_table or not stripped.startswith("| `"):
                continue
            key = stripped.split("`")[1].strip()
            if key and key not in known:
                violations.append(
                    f"docs/CONFIGURATION.md: key `{key}` is documented but absent from config.ini.example"
                )
        return violations

    def _context_window_coverage(self, example: dict[str, set[str]]) -> list[str]:
        providers = example.get("ai_providers", set())
        windows = example.get("model_context_windows", set())
        if not providers or not windows:
            return []
        selected = self._example_value("ai_providers", "provider") or "googleai"
        model_keys = {
            "googleai": ("google_studio_model",),
            "deepseek": ("deepseek_model",),
            "openrouter": ("openrouter_base_model",),
            "local": ("lm_studio_model",),
        }
        checked = {selected} if selected != "all" else {"googleai", "deepseek", "openrouter"}
        violations: list[str] = []
        for provider in sorted(checked):
            for model_key in model_keys.get(provider, ()):
                model = self._example_value("ai_providers", model_key)
                if not model:
                    continue
                target = f"{provider}/{model}"
                if target not in windows:
                    violations.append(
                        f"[model_context_windows] {target} is missing from config.ini.example; "
                        "the request would be refused before it is sent"
                    )
        return violations

    def _example_value(self, section: str, key: str) -> str:
        parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
        parser.optionxform = lambda option: option
        parser.read(self.example_path, encoding="utf-8")
        if not parser.has_option(section, key):
            return ""
        return parser.get(section, key).strip()

    def check(self) -> list[str]:

        example = self._read_ini(self.example_path)
        violations = self._missing_from_example(example)
        violations.extend(self._unknown_in_active_config(example))
        violations.extend(self._required_in_example(example))
        violations.extend(self._documented_keys(example))
        violations.extend(self._context_window_coverage(example))
        return violations


def main() -> int:
    violations = ConfigContractChecker().check()
    if not violations:
        print("config contract OK")
        return 0
    for violation in violations:
        print(violation)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
