# UCU AI Homework Report Template

Your report is published as a website in UCU colors. The page is generated from `report.md`, and it shows the date of your repository's last push.

## How to fill

1. Click **Use this template** to create your own **public** repository. If you fork instead, also open the **Actions** tab and enable workflows.
2. Enable GitHub Pages (see [below](#enabling-github-pages)).
3. Edit **`report.md`**:
   - Fill in the header block at the top (`title`, `author`, `course`, `date`).
   - Write the report in Markdown. You can use tables, code blocks, and LaTeX math (`$...$` inline, `$$...$$` for display).
   - Put images in `images/` and reference them as `![caption](images/plot.png)`.
4. Commit and push to `main`. GitHub Actions deploys the site automatically within about a minute, at `https://<username>.github.io/<repo>/`.
5. Submit that link.

## Enabling GitHub Pages

This is a one-time step. GitHub does not copy it to your repository.

1. In your repository, open **Settings → Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**. It saves as soon as you choose it.
3. Open the **Actions** tab, select the failed **Deploy report** run, and click **Re-run jobs**.

Until you do this, every deploy fails with the error "GitHub Pages is not set up".

## Local preview

```bash
python -m http.server 8000
# open http://localhost:8000
```

Opening `index.html` directly from disk will not work.

## ⚠️ Rules

You may change **only** `report.md` and the `images/` folder.

**Any change to the core website (`index.html`, `assets/`, `.github/`) will annul the grade for the homework.** This includes edits that change how the last-push date is shown.

The last-push date is read from GitHub and is used to check deadlines.
