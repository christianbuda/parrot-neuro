# Domain Description 1.1

Interfaces 4

Interface Scalp: scalp.tri
Interface Outer_skull: outer_skull.tri
Interface Inner_skull: inner_skull.tri
Interface Brain: brain.tri

Domains 5

Domain SCALP: -Scalp +Outer_skull
Domain BRAIN: -Brain
Domain CSF: -Inner_skull +Brain
Domain AIR: +Scalp
Domain SKULL: -Outer_skull +Inner_skull
