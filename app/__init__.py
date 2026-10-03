"""Laya AI Decision Service.

A reusable inference endpoint over the Laya decision model, built directly on
``laya.Router`` rather than on upstream's ``laya-serve``.
"""

from app.main import VERSION, build_default_app, create_app

__all__ = ["create_app", "build_default_app", "VERSION"]
__version__ = VERSION