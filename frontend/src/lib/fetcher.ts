export class FetcherError extends Error {
  info: unknown;
  status: number;

  constructor(message: string, status: number, info: unknown) {
    super(message);
    this.name = "FetcherError";
    this.status = status;
    this.info = info;
  }
}

export const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) {
    const info = await res.json().catch(() => null);
    throw new FetcherError(
      "An error occurred while fetching the data.",
      res.status,
      info,
    );
  }
  return res.json();
};
