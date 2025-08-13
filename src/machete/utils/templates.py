import jinja2
import tempfile
import os

def render_template(name: str, path: str, **kwargs) -> str:
    template_loader = jinja2.FileSystemLoader(searchpath=path)
    template_env = jinja2.Environment(loader=template_loader)

    tmp_dir = tempfile.gettempdir() + "/machete_tmp/" + name
    os.makedirs(tmp_dir, exist_ok=True)

    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            continue

        template = template_env.get_template(file)
        rendered_content = template.render(**kwargs)

        # Save to tmp directory
        tmp_file = os.path.join(tmp_dir, file)

        with open(tmp_file, 'w') as f:
            f.write(rendered_content)

    return tmp_dir
