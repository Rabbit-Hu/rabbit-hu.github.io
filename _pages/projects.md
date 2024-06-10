---
layout: archive
title: "Projects"
permalink: /projects/
author_profile: true
---

{% for post in site.projects reversed %}
{% include archive-single.html %}
{% endfor %}
