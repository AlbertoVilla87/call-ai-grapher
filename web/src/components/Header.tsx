function CircledWord({ children }: { children: string }) {
  return (
    <span className="circled">
      {children}
      <svg viewBox="0 0 220 90" aria-hidden="true" preserveAspectRatio="none">
        <path
          className="circled-path"
          pathLength="100"
          d="M28 52 C22 26, 92 12, 138 16 C186 20, 208 34, 204 50 C199 70, 128 82, 74 76 C36 71, 18 62, 24 44"
        />
      </svg>
    </span>
  );
}

export default function Header() {
  return (
    <header className="masthead rise">
      <div className="masthead-top">
        <p className="brand">
          <span className="nib" aria-hidden="true">
            ✒
          </span>
          CallAIgrapher
        </p>
        <p className="hand masthead-note">still yours.</p>
      </div>
      <h1 className="display headline">
        Your handwriting,{" "}
        <em>
          <CircledWord>refined</CircledWord>.
        </em>
      </h1>
      <p className="lede">
        Upload a scanned page, choose how far the ink travels toward a fair hand — and keep what is
        yours.
      </p>
    </header>
  );
}
