---
name: find-quda-installation
description: How to find quda installation
---

Follow these instructions to determine if and where QUDA is installed:
1. Use `nvidia-smi` to determine if there is any GPU. If there is no gpu, assume QUDA is not installed.
2. If GPU exisits, first check `/opt/quda/install`.
3. If QUDA does not exisit in `/opt/quda/install`, try to do your best to determine where QUDA lives.
4. Always ask the user explicitly if you cannot find QUDA but the system has GPU.
